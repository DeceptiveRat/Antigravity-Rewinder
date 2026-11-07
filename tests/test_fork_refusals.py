"""
Tests for the forks that must be refused, and the forks that must keep working.

A fork shares the child subagent conversation with its parent, so forking at a
subagent report would permanently change the child the parent still points to.
The TUI fork key also collects no intervention content, so it can only handle
prompts and model responses correctly.

Refusal tests compare every file in the session byte for byte, so any write is
caught, including writes to the child conversation.
"""

import json
import socket
import sqlite3
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from rewinding.Antigravity_Rewind import execute_fork, main
from rewinding.tui import RewindTUIApp

CONV = "11111111-1111-1111-1111-111111111111"
CHILD = "22222222-2222-2222-2222-222222222222"

FORK_REFUSAL = (
    "Forking at a subagent report is not supported yet. The fork would share the child "
    "subagent conversation with its parent, so the intervention would permanently change "
    "the child that the parent still points to. Use --out-dir to work on a full copy instead."
)
FORK_REFUSAL_BOX = f"\n{'!' * 80}\nError: {FORK_REFUSAL}\n{'!' * 80}\n\n"

# Every step type the TUI fork key must refuse, as (step index, detected category).
REFUSED_STEPS = [
    (1, "tool_call_read"),
    (2, "tool_result"),
    (3, "tool_call_command"),
    (5, "tool_call_write"),
    (7, "tool_call_web"),
    (9, "tool_call_inquiry"),
    (11, "tool_call_task"),
    (13, "tool_call_other"),
    (15, "subagent_report"),
    (17, "task_notification"),
    (18, "system_message"),
]

# Every step type it must still fork, as (step index, detected category, lineage label).
ALLOWED_STEPS = [
    (0, "prompt", ""),
    (16, "model_response", "before_response_regeneration"),
    (19, "prompt", ""),
    (20, "model_response", "before_response_regeneration"),
]


def tui_refusal(step: int, category: str) -> str:
    return f"Fork works only on prompts and model responses; step {step} is {category}."


# ---------------------------------------------------------------------------
# Protobuf encoding, written independently of the implementation under test
# ---------------------------------------------------------------------------

def varint(value: int) -> bytes:
    out = bytearray()
    while True:
        low_bits = value & 0x7F
        value >>= 7
        if value:
            out.append(low_bits | 0x80)
        else:
            out.append(low_bits)
            return bytes(out)


def bytes_field(number: int, data: bytes) -> bytes:
    return varint((number << 3) | 2) + varint(len(data)) + data


def int_field(number: int, value: int) -> bytes:
    return varint(number << 3) + varint(value)


def tool_result_payload(text: str) -> bytes:
    return bytes_field(14, bytes_field(4, text.encode()))


def system_message_payload(msg_id: str, sender: str, content: str, child_step: int | None = None) -> bytes:
    message = bytes_field(1, msg_id.encode()) + bytes_field(3, sender.encode()) + bytes_field(4, content.encode())
    if child_step is not None:
        message += bytes_field(12, bytes_field(1, bytes_field(1, sender.encode()) + int_field(2, child_step)))
    text = f"[Message] timestamp=2026-10-19T05:10:00Z sender={sender} priority=MESSAGE_PRIORITY_HIGH content={content}"
    return bytes_field(114, bytes_field(1, text.encode()) + bytes_field(4, message))


REPORT_TEXT = "Worker finished the analysis"
REPORT_PAYLOAD = system_message_payload("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", CHILD, REPORT_TEXT, child_step=2)
TASK_DONE_TEXT = f'Task id "{CONV}/task-4" finished with result:\n\nexit code 0'


# ---------------------------------------------------------------------------
# Session fixture
# ---------------------------------------------------------------------------

def create_conversation_db(path: Path, steps: list[tuple[int, int, bytes]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, status INTEGER, step_payload BLOB)")
    for table in ("gen_metadata", "executor_metadata", "parent_references", "battle_mode_infos"):
        conn.execute(f"CREATE TABLE {table} (idx INTEGER PRIMARY KEY, data BLOB)")
    for idx, step_type, payload in steps:
        conn.execute("INSERT INTO steps VALUES (?, ?, 3, ?)", (idx, step_type, payload))
        conn.execute("INSERT INTO parent_references VALUES (?, ?)", (idx, b"ref"))
        conn.execute("INSERT INTO battle_mode_infos VALUES (?, ?)", (idx, b"info"))
    for turn in range(sum(1 for _, step_type, _ in steps if step_type == 15)):
        conn.execute("INSERT INTO gen_metadata VALUES (?, ?)", (turn, b"gen"))
        conn.execute("INSERT INTO executor_metadata VALUES (?, ?)", (turn, b"exec"))
    conn.commit()
    conn.close()


def write_transcripts(system_generated: Path, records: list[dict]) -> None:
    text = "".join(json.dumps(record) + "\n" for record in records)
    logs = system_generated / "logs"
    for name in ("transcript_full", "transcript"):
        chunks = logs / "chunks" / name
        chunks.mkdir(parents=True)
        (logs / f"{name}.jsonl").write_text(text, encoding="utf-8")
        (chunks / "00000000.jsonl").write_text(text, encoding="utf-8")


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def refuse_connection(*args, **kwargs):
        raise AssertionError("A fork must never open a network connection.")

    monkeypatch.setattr(socket.socket, "connect", refuse_connection)


@pytest.fixture
def base_dir(tmp_path) -> Path:
    """A session containing every step type, plus a child subagent conversation with 6 steps."""
    base = tmp_path / ".gemini" / "antigravity-cli"
    (base / "conversations").mkdir(parents=True)

    def tool_call(name: str) -> dict:
        return {"type": "PLANNER_RESPONSE", "source": "MODEL", "tool_calls": [{"name": name, "args": {}}]}

    def tool_result(text: str) -> tuple[bytes, dict]:
        return tool_result_payload(text), {"type": "TOOL_RESULT", "source": "MODEL", "content": text}

    def prompt(text: str) -> dict:
        return {"type": "USER_INPUT", "source": "USER_EXPLICIT", "content": text}

    def reply(text: str) -> dict:
        return {"type": "PLANNER_RESPONSE", "source": "MODEL", "content": text}

    report_line = (
        f"<SYSTEM_MESSAGE>\n[Message] timestamp=2026-10-19T05:10:00Z sender={CHILD} "
        f"priority=MESSAGE_PRIORITY_HIGH content={REPORT_TEXT}\n</SYSTEM_MESSAGE>"
    )
    rows = [
        (14, b"first prompt", prompt("first prompt")),
        (15, b"planner", tool_call("view_file")),
        (132, *tool_result("ORIGINAL FILE TEXT")),
        (15, b"planner", tool_call("run_command")),
        (132, *tool_result("command output")),
        (15, b"planner", tool_call("write_to_file")),
        (132, *tool_result("file written")),
        (15, b"planner", tool_call("read_url_content")),
        (132, *tool_result("page saved")),
        (15, b"planner", tool_call("ask_question")),
        (132, *tool_result("user answered")),
        (15, b"planner", tool_call("manage_task")),
        (132, *tool_result("task running")),
        (15, b"planner", tool_call("invoke_subagent")),
        (132, *tool_result("subagent spawned")),
        (101, REPORT_PAYLOAD, {"type": "SYSTEM_MESSAGE", "source": "SYSTEM", "content": report_line}),
        (15, b"planner", reply("subagent finished")),
        (
            101,
            system_message_payload("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", f"{CONV}/task-4", TASK_DONE_TEXT),
            {"type": "SYSTEM_MESSAGE", "source": "SYSTEM", "content": TASK_DONE_TEXT},
        ),
        (
            101,
            system_message_payload("cccccccc-cccc-cccc-cccc-cccccccccccc", "system", "platform notice"),
            {"type": "SYSTEM_MESSAGE", "source": "SYSTEM", "content": "platform notice"},
        ),
        (14, b"second prompt", prompt("second prompt")),
        (15, b"planner", reply("all done")),
    ]
    create_conversation_db(
        base / "conversations" / f"{CONV}.db",
        [(idx, step_type, payload) for idx, (step_type, payload, _) in enumerate(rows)],
    )
    write_transcripts(
        base / "brain" / CONV / ".system_generated",
        [
            {"step_index": idx, "created_at": f"2026-10-19T05:10:{idx:02d}Z", **record}
            for idx, (_, _, record) in enumerate(rows)
        ],
    )

    create_conversation_db(
        base / "conversations" / f"{CHILD}.db",
        [(idx, 15 if idx % 2 else 14, b"child step") for idx in range(6)],
    )
    write_transcripts(
        base / "brain" / CHILD / ".system_generated",
        [
            {"step_index": idx, "type": "GENERIC", "source": "MODEL", "content": f"child {idx}",
             "created_at": f"2026-10-19T05:09:{idx:02d}Z"}
            for idx in range(6)
        ],
    )

    conn = sqlite3.connect(base / "conversation_summaries.db")
    conn.execute("""
        CREATE TABLE conversation_summaries (
            conversation_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            preview TEXT NOT NULL DEFAULT '',
            step_count INTEGER NOT NULL DEFAULT 0,
            last_modified_time TEXT,
            workspace_uris TEXT,
            status TEXT NOT NULL DEFAULT 'CASCADE_RUN_STATUS_IDLE',
            parent_conversation_id TEXT NOT NULL DEFAULT '',
            nesting_depth INTEGER NOT NULL DEFAULT 0,
            not_fully_idle INTEGER NOT NULL DEFAULT 0,
            killed INTEGER NOT NULL DEFAULT 0,
            last_user_input_time TEXT,
            last_user_input_step_index INTEGER NOT NULL DEFAULT -1,
            raw_summary BLOB
        )
    """)
    conn.execute(
        "INSERT INTO conversation_summaries (conversation_id, title, step_count, last_user_input_step_index) "
        "VALUES (?, 'Parent', 21, 19)",
        (CONV,),
    )
    conn.execute(
        "INSERT INTO conversation_summaries (conversation_id, title, step_count, parent_conversation_id, nesting_depth) "
        "VALUES (?, 'Child', 6, ?, 1)",
        (CHILD, CONV),
    )
    conn.commit()
    conn.close()

    (base / "history.jsonl").write_text(
        json.dumps({"conversationId": CONV, "display": "first prompt", "timestamp": 1792386600000}) + "\n",
        encoding="utf-8",
    )
    return base


# ---------------------------------------------------------------------------
# State inspection
# ---------------------------------------------------------------------------

def snapshot(root: Path) -> dict[str, bytes]:
    """Every file under root, mapped to its exact bytes."""
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def step_indices(base: Path, conv_id: str) -> list[int]:
    with sqlite3.connect(base / "conversations" / f"{conv_id}.db") as conn:
        return [row[0] for row in conn.execute("SELECT idx FROM steps ORDER BY idx")]


def step_payload(base: Path, conv_id: str, idx: int) -> bytes:
    with sqlite3.connect(base / "conversations" / f"{conv_id}.db") as conn:
        return conn.execute("SELECT step_payload FROM steps WHERE idx = ?", (idx,)).fetchone()[0]


def catalog_row(base: Path, conv_id: str) -> tuple:
    with sqlite3.connect(base / "conversation_summaries.db") as conn:
        return conn.execute("SELECT * FROM conversation_summaries WHERE conversation_id = ?", (conv_id,)).fetchone()


def lineage(base: Path) -> list[tuple]:
    with sqlite3.connect(base / "conversation_summaries.db") as conn:
        return conn.execute(
            "SELECT fork_id, parent_id, fork_step, branch_name, intervention_type FROM conversation_lineage"
        ).fetchall()


def originals(base: Path, fork_id: str | None = None) -> dict:
    """
    The state a successful fork must leave alone: every file except the catalog and
    the fork's own files, plus the parent's and child's rows in the catalog.
    """
    files = snapshot(base)
    del files["conversation_summaries.db"]
    if fork_id:
        files = {
            name: data for name, data in files.items()
            if name != f"conversations/{fork_id}.db" and not name.startswith(f"brain/{fork_id}/")
        }
    return {"files": files, "parent_row": catalog_row(base, CONV), "child_row": catalog_row(base, CHILD)}


def run_cli(monkeypatch, base: Path, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["agy-rewind", "-d", str(base), "-c", CONV, *args])
    main()


def assert_cli_refuses_fork(monkeypatch, capsys, base: Path, *args: str) -> None:
    before = snapshot(base)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base, *args)
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == FORK_REFUSAL_BOX
    assert snapshot(base) == before


def open_tui(monkeypatch, base: Path, cursor: int, dry_run: bool = False) -> tuple[RewindTUIApp, list[str]]:
    """Opens the TUI at a step. Only the terminal dialogs are replaced, since they need a real terminal."""
    app = RewindTUIApp(str(base), CONV, dry_run=dry_run)
    app.cursor = cursor
    questions: list[str] = []

    def answer_branch_name(question: str) -> str:
        questions.append(question)
        return "b"

    monkeypatch.setattr(app, "_show_input_dialog", answer_branch_name)
    monkeypatch.setattr(app, "_show_confirm_dialog", lambda question: False)
    return app, questions


# ---------------------------------------------------------------------------
# Command line: forks at subagent reports are refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("options", [
    ["--fork", "b"],
    ["--fork", "b", "--subagent-exit", "crash"],
    ["--fork", "b", "--subagent-exit", "complete"],
    ["--fork", "b", "--subagent-exit", "keep-alive", "--replace-text", "edited report"],
    ["--fork", "b", "--dry-run"],
])
def test_cli_refuses_fork_at_subagent_report_in_every_mode(base_dir, monkeypatch, capsys, options):
    assert_cli_refuses_fork(monkeypatch, capsys, base_dir, "15", *options)


@pytest.mark.parametrize("branch", ["", "   ", "a\x00b", "../../escape", "-looks-like-a-flag", "x" * 10_000])
def test_cli_refusal_does_not_depend_on_branch_name(base_dir, monkeypatch, capsys, branch):
    assert_cli_refuses_fork(monkeypatch, capsys, base_dir, "15", f"--fork={branch}")


@pytest.mark.parametrize("damaged_payload", [
    b"",
    b"   ",
    b"\x00",
    b"\x00\xff\x00",
    REPORT_PAYLOAD[:20],
])
def test_cli_refuses_fork_when_report_payload_is_damaged(base_dir, monkeypatch, capsys, damaged_payload):
    """The transcript still names the sender, so the step must still be recognised as a report."""
    with sqlite3.connect(base_dir / "conversations" / f"{CONV}.db") as conn:
        conn.execute("UPDATE steps SET step_payload = ? WHERE idx = 15", (damaged_payload,))
    assert_cli_refuses_fork(monkeypatch, capsys, base_dir, "15", "--fork", "b")


def test_cli_refuses_fork_when_transcript_has_a_malformed_record(base_dir, monkeypatch, capsys):
    """Valid records mixed with a malformed one must neither crash the check nor bypass it."""
    with sqlite3.connect(base_dir / "conversations" / f"{CONV}.db") as conn:
        conn.execute("UPDATE steps SET step_payload = ? WHERE idx = 15", (b"",))
    transcript = base_dir / "brain" / CONV / ".system_generated" / "logs" / "transcript_full.jsonl"
    transcript.write_text('{"step_index": 15, "content": \n' + transcript.read_text(encoding="utf-8"), encoding="utf-8")
    assert_cli_refuses_fork(monkeypatch, capsys, base_dir, "15", "--fork", "b")


def test_cli_fork_refusal_comes_before_the_task_log_check(base_dir, monkeypatch, capsys):
    """A later task completion must not send the user down the task-log options for a fork that cannot happen."""
    tasks = base_dir / "brain" / CONV / ".system_generated" / "tasks"
    tasks.mkdir()
    (tasks / "task-4.log").write_text("FUTURE OUTPUT\n", encoding="utf-8")
    assert_cli_refuses_fork(monkeypatch, capsys, base_dir, "15", "--fork", "b")


def test_out_dir_alternative_named_in_the_refusal_leaves_the_original_alone(base_dir, tmp_path, monkeypatch):
    """The refusal recommends --out-dir, so it must really keep the original child safe."""
    before = snapshot(base_dir)
    copy_dir = tmp_path / "copy"
    run_cli(monkeypatch, base_dir, "15", "--subagent-exit", "crash", "--out-dir", str(copy_dir))
    assert snapshot(base_dir) == before
    assert step_indices(copy_dir, CHILD) == [0, 1, 2]


# ---------------------------------------------------------------------------
# execute_fork itself: the protection holds for every caller
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dry_run", [False, True])
def test_execute_fork_raises_at_subagent_report(base_dir, dry_run):
    before = snapshot(base_dir)
    with pytest.raises(ValueError) as error:
        execute_fork(str(base_dir), CONV, 15, branch_name="b", dry_run=dry_run, category="subagent_report")
    assert str(error.value) == FORK_REFUSAL
    assert snapshot(base_dir) == before


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(branch=st.text(max_size=200), dry_run=st.booleans())
def test_execute_fork_refusal_holds_for_any_branch_name(base_dir, branch, dry_run):
    """Invariant: no branch name or dry-run setting changes the refusal or lets a write through."""
    before = snapshot(base_dir)
    with pytest.raises(ValueError) as error:
        execute_fork(str(base_dir), CONV, 15, branch_name=branch, dry_run=dry_run, category="subagent_report")
    assert str(error.value) == FORK_REFUSAL
    assert snapshot(base_dir) == before


# ---------------------------------------------------------------------------
# Command line: other forks keep working
# ---------------------------------------------------------------------------

def test_cli_fork_at_prompt_still_works(base_dir, monkeypatch):
    before = originals(base_dir)
    run_cli(monkeypatch, base_dir, "19", "--fork", "b")
    [(fork_id, parent_id, fork_step, branch, label)] = lineage(base_dir)
    assert (parent_id, fork_step, branch, label) == (CONV, 19, "b", "")
    assert step_indices(base_dir, fork_id) == list(range(19))
    assert originals(base_dir, fork_id) == before


def test_cli_fork_at_tool_call_still_rewrites_only_the_fork(base_dir, monkeypatch):
    before = originals(base_dir)
    run_cli(monkeypatch, base_dir, "1", "--replace-text", "COUNTERFACTUAL TEXT", "--fork", "b")
    [(fork_id, parent_id, fork_step, branch, label)] = lineage(base_dir)
    assert (parent_id, fork_step, branch, label) == (CONV, 1, "b", "tool_content_replacement")
    assert step_indices(base_dir, fork_id) == [0, 1, 2]
    assert step_payload(base_dir, fork_id, 2) == tool_result_payload("COUNTERFACTUAL TEXT")
    assert originals(base_dir, fork_id) == before


# ---------------------------------------------------------------------------
# TUI: the fork key refuses steps it cannot handle
# ---------------------------------------------------------------------------

def test_fixture_covers_every_expected_step_type(base_dir):
    """Guards the tests below: a fixture drift could otherwise make them pass without testing the right steps."""
    app = RewindTUIApp(str(base_dir), CONV)
    detected = [(step.idx, step.category) for step in app.loader.steps]
    expected = {idx: category for idx, category in REFUSED_STEPS}
    expected.update({idx: category for idx, category, _ in ALLOWED_STEPS})
    assert [pair for pair in detected if pair[0] in expected] == sorted(expected.items())


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("step, category", REFUSED_STEPS)
def test_tui_fork_action_refuses_unsupported_step(base_dir, monkeypatch, step, category, dry_run):
    before = snapshot(base_dir)
    app, _ = open_tui(monkeypatch, base_dir, step, dry_run=dry_run)
    expected = tui_refusal(step, category)
    assert app.execute_action_fork("b") == (False, expected, None)
    assert app.status_msg == expected
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("step, category", REFUSED_STEPS)
def test_tui_fork_key_refuses_before_asking_for_a_branch_name(base_dir, monkeypatch, step, category):
    before = snapshot(base_dir)
    app, questions = open_tui(monkeypatch, base_dir, step)
    app.handle_input_char(ord("f"))
    assert questions == []
    assert app.status_msg == tui_refusal(step, category)
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("cursor", [-1, 21, 2 ** 63])
def test_tui_fork_with_out_of_range_cursor_does_nothing(base_dir, monkeypatch, cursor):
    before = snapshot(base_dir)
    app, _ = open_tui(monkeypatch, base_dir, cursor)
    assert app.execute_action_fork("b") == (False, "No step selected.", None)
    assert snapshot(base_dir) == before


# ---------------------------------------------------------------------------
# TUI: the fork key keeps working on prompts and model responses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step, category, label", ALLOWED_STEPS)
def test_tui_fork_key_still_forks_prompts_and_model_responses(base_dir, monkeypatch, step, category, label):
    before = originals(base_dir)
    app, questions = open_tui(monkeypatch, base_dir, step)
    assert app.get_selected_step().category == category
    app.handle_input_char(ord("f"))
    [(fork_id, parent_id, fork_step, branch, recorded_label)] = lineage(base_dir)
    assert (parent_id, fork_step, branch, recorded_label) == (CONV, step, "b", label)
    assert questions == [f"Fork @ Step {step} - Branch Name (default: 'branch'):"]
    assert app.status_msg == f"Forked to branch 'b' ({fork_id[:8]}...)."
    assert step_indices(base_dir, fork_id) == list(range(step))
    assert originals(base_dir, fork_id) == before


@pytest.mark.parametrize("step, category, label", ALLOWED_STEPS)
def test_tui_fork_dry_run_creates_nothing(base_dir, monkeypatch, step, category, label):
    before = snapshot(base_dir)
    app, _ = open_tui(monkeypatch, base_dir, step, dry_run=True)
    success, message, fork_id = app.execute_action_fork("b")
    assert (success, message) == (True, f"Forked to branch 'b' ({fork_id[:8]}...) [DRY-RUN].")
    assert snapshot(base_dir) == before
