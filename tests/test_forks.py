"""
Tests for conversation forks: which forks are refused, what a successful fork
produces, and that the original session is never changed.

Also covers the TUI's question about task logs, because rewinding to an earlier
step leaves those logs holding output from steps that no longer exist.

Refusal tests compare every file in the session byte for byte, so any write is
caught, including writes to the child subagent conversation.
"""

import json
import shutil
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

LOG_2 = "TASK 2 FINAL OUTPUT\n"
LOG_4 = "TASK 4 FINAL OUTPUT\n"
NEW_TEXT = "NEW TEXT"

FORK_REFUSAL = (
    "Forking at a subagent report is not supported yet. The fork would share the child "
    "subagent conversation with its parent, so the intervention would permanently change "
    "the child that the parent still points to. Use --out-dir to work on a full copy instead."
)
UNDECIDED_LOGS = "Task logs hold output from steps this fork discards: task-4, task-2. Decide what they should contain."

# Steps the TUI fork key must refuse, as (step index, detected category, status message).
REFUSED_STEPS = [
    (2, "tool_result", "Cannot fork at a tool result; select the tool call at step 1."),
    (13, "tool_call_other", "Cannot fork at step 13: results of invoke_subagent cannot be changed."),
    (15, "subagent_report", "Cannot fork at a subagent report yet; it would change the shared subagent."),
    (18, "system_message", "Cannot fork at step 18: system_message steps cannot be changed."),
]
TOOL_CALL_STEPS = [
    (1, "tool_call_read"),
    (3, "tool_call_command"),
    (5, "tool_call_write"),
    (7, "tool_call_web"),
    (9, "tool_call_inquiry"),
    (11, "tool_call_task"),
]
NOTIFICATION_STEPS = [(17, "task_notification"), (21, "task_notification")]
PROMPT_STEPS = [(0, "prompt"), (19, "prompt")]
REPLY_STEPS = [(16, "model_response"), (20, "model_response")]


def error_box(message: str) -> str:
    return f"\n{'!' * 80}\nError: {message}\n{'!' * 80}\n\n"


def log_question(count: int) -> str:
    return f"{count} task log(s) hold future output. Clear them? [y/N]: "


def logs_kept_message(count: int) -> str:
    return f"Cancelled: {count} task log(s) would keep future output."


def branch_question(step: int) -> str:
    return f"Fork @ Step {step} - Branch Name (default: 'branch'):"


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
TASK_2_DONE = f'Task id "{CONV}/task-2" finished with result:\n\nexit code 0'
TASK_4_DONE = f'Task id "{CONV}/task-4" finished with result:\n\nexit code 0'


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
    """
    A session containing every step type, plus a child subagent conversation with 6 steps.
    Task completions appear at steps 17 (task-4) and 21 (task-2). Their logs only exist
    after add_task_logs, so tests opt in to the future-log checks.
    """
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

    def notification(msg_id: str, sender: str, content: str) -> tuple[bytes, dict]:
        return (
            system_message_payload(msg_id, sender, content),
            {"type": "SYSTEM_MESSAGE", "source": "SYSTEM", "content": content},
        )

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
        (101, *notification("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", f"{CONV}/task-4", TASK_4_DONE)),
        (101, *notification("cccccccc-cccc-cccc-cccc-cccccccccccc", "system", "platform notice")),
        (14, b"second prompt", prompt("second prompt")),
        (15, b"planner", reply("all done")),
        (101, *notification("dddddddd-dddd-dddd-dddd-dddddddddddd", f"{CONV}/task-2", TASK_2_DONE)),
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
        "VALUES (?, 'Parent', 22, 19)",
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


def add_task_logs(base: Path) -> None:
    tasks = base / "brain" / CONV / ".system_generated" / "tasks"
    tasks.mkdir(exist_ok=True)
    (tasks / "task-2.log").write_text(LOG_2, encoding="utf-8")
    (tasks / "task-4.log").write_text(LOG_4, encoding="utf-8")


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


def run_state(base: Path, conv_id: str) -> tuple[str, int]:
    with sqlite3.connect(base / "conversation_summaries.db") as conn:
        return conn.execute(
            "SELECT status, not_fully_idle FROM conversation_summaries WHERE conversation_id = ?", (conv_id,)
        ).fetchone()


def lineage(base: Path) -> list[tuple]:
    with sqlite3.connect(base / "conversation_summaries.db") as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'conversation_lineage'").fetchone():
            return []
        return conn.execute(
            "SELECT fork_id, parent_id, fork_step, branch_name, intervention_type FROM conversation_lineage"
        ).fetchall()


def task_logs(base: Path, conv_id: str) -> dict[str, str]:
    folder = base / "brain" / conv_id / ".system_generated" / "tasks"
    return {path.stem: path.read_text(encoding="utf-8") for path in folder.glob("*.log")}


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


def assert_no_fork_exists(base: Path) -> None:
    assert sorted(path.stem for path in (base / "conversations").iterdir()) == [CONV, CHILD]
    assert sorted(path.name for path in (base / "brain").iterdir()) == [CONV, CHILD]
    with sqlite3.connect(base / "conversation_summaries.db") as conn:
        rows = conn.execute("SELECT conversation_id FROM conversation_summaries ORDER BY conversation_id").fetchall()
    assert rows == [(CONV,), (CHILD,)]
    assert lineage(base) == []


def run_cli(monkeypatch, base: Path, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["agy-rewind", "-d", str(base), "-c", CONV, *args])
    main()


def assert_cli_refuses_fork(monkeypatch, capsys, base: Path, *args: str) -> None:
    before = snapshot(base)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base, *args)
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == error_box(FORK_REFUSAL)
    assert snapshot(base) == before


def open_tui(monkeypatch, base: Path, cursor: int, dry_run: bool = False, answers=("b",), clear_logs: bool = False):
    """
    Opens the TUI at a step and answers its dialogs from a script. Only the dialogs
    are replaced, since they need a real terminal. Returns the app and the questions asked.
    """
    app = RewindTUIApp(str(base), CONV, dry_run=dry_run)
    app.cursor = cursor
    inputs: list[str] = []
    confirms: list[str] = []
    remaining = list(answers)

    def answer_input(question: str) -> str | None:
        inputs.append(question)
        return remaining.pop(0)

    def answer_confirm(question: str) -> bool:
        confirms.append(question)
        if question.startswith("Rewind"):
            return True
        return clear_logs and "task log(s)" in question

    monkeypatch.setattr(app, "_show_input_dialog", answer_input)
    monkeypatch.setattr(app, "_show_confirm_dialog", answer_confirm)
    return app, inputs, confirms


def only_fork(base: Path) -> tuple:
    [row] = lineage(base)
    return row


# ---------------------------------------------------------------------------
# Forks at subagent reports are refused
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
    add_task_logs(base_dir)
    assert_cli_refuses_fork(monkeypatch, capsys, base_dir, "15", "--fork", "b")


def test_out_dir_alternative_named_in_the_refusal_leaves_the_original_alone(base_dir, tmp_path, monkeypatch):
    """The refusal recommends --out-dir, so it must really keep the original child safe."""
    before = snapshot(base_dir)
    copy_dir = tmp_path / "copy"
    run_cli(monkeypatch, base_dir, "15", "--subagent-exit", "crash", "--out-dir", str(copy_dir))
    assert snapshot(base_dir) == before
    assert step_indices(copy_dir, CHILD) == [0, 1, 2]


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
# execute_fork checks a request before copying anything
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", ["tool_result", "tool_call_other", "system_message", "unknown", "", "prompt "])
def test_execute_fork_refuses_steps_it_has_no_handler_for(base_dir, category):
    """Without this check, these categories produced an identical copy reported as a success."""
    before = snapshot(base_dir)
    with pytest.raises(ValueError) as error:
        execute_fork(str(base_dir), CONV, 1, branch_name="b", category=category)
    assert str(error.value) == f"Cannot fork at a step of type '{category}'."
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("step, category", TOOL_CALL_STEPS + NOTIFICATION_STEPS)
@pytest.mark.parametrize("content", [{}, {"replace_text": None}, {"replace_text": ""}])
def test_execute_fork_requires_intervention_content(base_dir, step, category, content):
    """Without this check, a manage_task fork blanked the result and a notification fork blanked the next step."""
    before = snapshot(base_dir)
    with pytest.raises(ValueError) as error:
        execute_fork(str(base_dir), CONV, step, branch_name="b", category=category, intervention_kwargs=content)
    assert str(error.value) == f"Forking at a step of type '{category}' requires intervention content."
    assert snapshot(base_dir) == before


def test_zero_exit_code_counts_as_intervention_content(base_dir):
    """Zero is a clean exit, not a missing value, so a truthiness check would wrongly refuse it."""
    fork_id = execute_fork(str(base_dir), CONV, 11, branch_name="b", category="tool_call_task",
                           intervention_kwargs={"task_exit_code": 0})
    assert step_indices(base_dir, fork_id) == list(range(13))


def test_default_simulated_error_counts_as_intervention_content(base_dir, monkeypatch):
    """--simulate-error with no value is stored as an empty string, which must not look like missing content."""
    run_cli(monkeypatch, base_dir, "3", "--simulate-error", "--fork", "b")
    fork_id, parent_id, fork_step, branch, label = only_fork(base_dir)
    assert (parent_id, fork_step, branch, label) == (CONV, 3, "b", "simulate_error=")
    assert step_indices(base_dir, fork_id) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("existing", [CONV, CHILD])
def test_execute_fork_never_overwrites_an_existing_conversation(base_dir, existing):
    """A failed fork deletes its destination, so an existing conversation as destination would be destroyed."""
    before = snapshot(base_dir)
    with pytest.raises(ValueError) as error:
        execute_fork(str(base_dir), CONV, 19, branch_name="b", category="prompt", fork_conv_id=existing)
    assert str(error.value) == f"Fork destination already exists: {existing}"
    assert snapshot(base_dir) == before


# ---------------------------------------------------------------------------
# A failed fork leaves nothing behind
# ---------------------------------------------------------------------------

def test_cli_fork_failure_reports_cleanly_and_removes_the_copy(base_dir, monkeypatch, capsys):
    before = originals(base_dir)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, "1", "--replace-file", "/nonexistent/file", "--fork", "b")
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == error_box("Replacement file not found: /nonexistent/file")
    assert_no_fork_exists(base_dir)
    assert originals(base_dir) == before


@pytest.mark.parametrize("damage, message", [
    ("DELETE FROM steps WHERE idx = 2", "Result step 2 does not exist in DB for tool call at step 1."),
    ("UPDATE steps SET step_type = 15 WHERE idx = 2", "Step 2 has step_type 15, expected 132 or 5 (tool result for step 1)."),
])
def test_execute_fork_removes_the_copy_when_the_engine_fails(base_dir, damage, message):
    with sqlite3.connect(base_dir / "conversations" / f"{CONV}.db") as conn:
        conn.execute(damage)
    before = originals(base_dir)
    with pytest.raises(ValueError) as error:
        execute_fork(str(base_dir), CONV, 1, branch_name="b", category="tool_call_read",
                     intervention_kwargs={"tool_name": "view_file", "replace_text": NEW_TEXT})
    assert str(error.value) == message
    assert_no_fork_exists(base_dir)
    assert originals(base_dir) == before


def test_tui_fork_failure_shows_the_reason_and_removes_the_copy(base_dir, monkeypatch):
    app, _, _ = open_tui(monkeypatch, base_dir, 1, answers=(NEW_TEXT, "b"))
    with sqlite3.connect(base_dir / "conversations" / f"{CONV}.db") as conn:
        conn.execute("DELETE FROM steps WHERE idx = 2")
    before = originals(base_dir)
    app.handle_input_char(ord("f"))
    assert app.status_msg == "Fork failed: Result step 2 does not exist in DB for tool call at step 1."
    assert_no_fork_exists(base_dir)
    assert originals(base_dir) == before


# ---------------------------------------------------------------------------
# Command line: successful forks
# ---------------------------------------------------------------------------

def test_cli_fork_at_prompt_still_works(base_dir, monkeypatch):
    before = originals(base_dir)
    run_cli(monkeypatch, base_dir, "19", "--fork", "b")
    fork_id, parent_id, fork_step, branch, label = only_fork(base_dir)
    assert (parent_id, fork_step, branch, label) == (CONV, 19, "b", "")
    assert step_indices(base_dir, fork_id) == list(range(19))
    assert originals(base_dir, fork_id) == before


def test_cli_fork_at_tool_call_still_rewrites_only_the_fork(base_dir, monkeypatch):
    before = originals(base_dir)
    run_cli(monkeypatch, base_dir, "1", "--replace-text", NEW_TEXT, "--fork", "b")
    fork_id, parent_id, fork_step, branch, label = only_fork(base_dir)
    assert (parent_id, fork_step, branch, label) == (CONV, 1, "b", "tool_content_replacement")
    assert step_indices(base_dir, fork_id) == [0, 1, 2]
    assert step_payload(base_dir, fork_id, 2) == tool_result_payload(NEW_TEXT)
    assert originals(base_dir, fork_id) == before


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(text=st.text(min_size=1, max_size=300))
def test_replacement_text_round_trips_through_the_fork_payload(base_dir, text):
    """Round trip: any text, including nulls and multi-byte characters, must decode back from valid Protobuf."""
    before = originals(base_dir)
    fork_id = execute_fork(str(base_dir), CONV, 1, branch_name="b", category="tool_call_read",
                           intervention_kwargs={"tool_name": "view_file", "replace_text": text})
    assert step_payload(base_dir, fork_id, 2) == tool_result_payload(text)
    assert originals(base_dir, fork_id) == before


# ---------------------------------------------------------------------------
# Command line: forks write the task log content that was requested
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("options, expected", [
    (["--task-output", ""], {"task-2": "", "task-4": ""}),
    (["--task-output", "partial"], {"task-2": "partial", "task-4": "partial"}),
    (["--task-output", "   "], {"task-2": "   ", "task-4": "   "}),
    (["--task-output", "a\x00b"], {"task-2": "a\x00b", "task-4": "a\x00b"}),
    (["--task-output", "한글 로그"], {"task-2": "한글 로그", "task-4": "한글 로그"}),
    (["--task-output", "task-2=", "--task-output", "task-4=partial"], {"task-2": "", "task-4": "partial"}),
    (["--keep-task-log"], {"task-2": LOG_2, "task-4": LOG_4}),
    (["--keep-task-log", "--task-output", "task-2="], {"task-2": "", "task-4": LOG_4}),
])
def test_fork_writes_the_requested_log_content(base_dir, monkeypatch, options, expected):
    add_task_logs(base_dir)
    before = originals(base_dir)
    run_cli(monkeypatch, base_dir, "16", *options, "--fork", "b")
    fork_id = only_fork(base_dir)[0]
    assert task_logs(base_dir, fork_id) == expected
    assert originals(base_dir, fork_id) == before


def test_fork_loads_log_content_from_a_file(base_dir, tmp_path, monkeypatch):
    add_task_logs(base_dir)
    source = tmp_path / "log_source.txt"
    source.write_text("FROM A FILE\n", encoding="utf-8")
    run_cli(monkeypatch, base_dir, "16", "--task-output", str(source), "--fork", "b")
    assert task_logs(base_dir, only_fork(base_dir)[0]) == {"task-2": "FROM A FILE\n", "task-4": "FROM A FILE\n"}


def test_fork_leaves_logs_of_tasks_that_finished_before_the_fork_point(base_dir, monkeypatch):
    """task-4 finished at step 17, before the fork at step 19, so its log is not future output."""
    add_task_logs(base_dir)
    run_cli(monkeypatch, base_dir, "19", "--task-output", "", "--fork", "b")
    assert task_logs(base_dir, only_fork(base_dir)[0]) == {"task-2": "", "task-4": LOG_4}


@pytest.mark.parametrize("step, options", [
    (16, []),
    (19, []),
    (3, ["--replace-text", NEW_TEXT]),
    (5, ["--replace-text", NEW_TEXT]),
    (7, ["--replace-text", NEW_TEXT]),
    (9, ["--replace-text", NEW_TEXT]),
    (11, ["--replace-text", NEW_TEXT]),
    (17, ["--task-status", "failed"]),
])
@pytest.mark.parametrize("log_options", [
    ["--task-output", ""],
    ["--task-output", "partial"],
    ["--task-output", "task-2=x", "--task-output", "task-4=y"],
    ["--keep-task-log"],
])
def test_fork_and_in_place_rewind_leave_identical_logs(base_dir, tmp_path, monkeypatch, step, options, log_options):
    """Oracle: the same request must produce the same logs whether or not it runs in a fork."""
    add_task_logs(base_dir)
    in_place = tmp_path / "in_place"
    shutil.copytree(base_dir, in_place)
    before = originals(base_dir)

    run_cli(monkeypatch, in_place, str(step), *options, *log_options)
    run_cli(monkeypatch, base_dir, str(step), *options, *log_options, "--fork", "b")

    fork_id = only_fork(base_dir)[0]
    assert task_logs(base_dir, fork_id) == task_logs(in_place, CONV)
    assert originals(base_dir, fork_id) == before


def test_fork_dry_run_previews_log_updates_and_writes_nothing(base_dir, monkeypatch, capsys):
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    run_cli(monkeypatch, base_dir, "16", "--task-output", "partial", "--fork", "b", "--dry-run")
    output = capsys.readouterr().out.splitlines()
    assert "[DRY-RUN] Would update task log: tasks/task-4.log (7 bytes)" in output
    assert "[DRY-RUN] Would update task log: tasks/task-2.log (7 bytes)" in output
    assert snapshot(base_dir) == before


def test_cli_fork_is_refused_until_logs_are_decided(base_dir, monkeypatch, capsys):
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, "16", "--fork", "b")
    assert exit_info.value.code == 1
    assert "Rewind cuts off downstream completion of multiple background tasks" in capsys.readouterr().err
    assert snapshot(base_dir) == before


def test_execute_fork_itself_refuses_undecided_logs(base_dir):
    """Protects any caller that skips the command line's check."""
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    with pytest.raises(ValueError) as error:
        execute_fork(str(base_dir), CONV, 16, branch_name="b", category="model_response")
    assert str(error.value) == UNDECIDED_LOGS
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("fork_options", [[], ["--fork", "b"]])
def test_replacement_text_survives_a_log_option_on_a_task_tool_call(base_dir, monkeypatch, fork_options):
    """The task engine used to drop --replace-text when a log option was present, which blanked the result."""
    add_task_logs(base_dir)
    run_cli(monkeypatch, base_dir, "11", "--replace-text", NEW_TEXT, "--task-output", "", *fork_options)
    conv_id = only_fork(base_dir)[0] if fork_options else CONV
    assert step_payload(base_dir, conv_id, 12) == tool_result_payload(NEW_TEXT)
    assert task_logs(base_dir, conv_id) == {"task-2": "", "task-4": ""}


@pytest.mark.parametrize("fork_options", [[], ["--fork", "b"]])
def test_replacement_text_survives_a_log_option_on_a_notification(base_dir, monkeypatch, fork_options):
    add_task_logs(base_dir)
    run_cli(monkeypatch, base_dir, "17", "--replace-text", NEW_TEXT, "--task-output", "", *fork_options)
    conv_id = only_fork(base_dir)[0] if fork_options else CONV
    payload = step_payload(base_dir, conv_id, 17)
    assert NEW_TEXT.encode() in payload
    assert TASK_4_DONE.encode() not in payload


@pytest.mark.parametrize("output_file_text, original_text", [
    # output.txt differs from the transcript, so finding this text proves output.txt was read
    ("text only in output.txt", "text only in output.txt"),
    # with no output.txt, the transcript is the only remaining source
    (None, "command output"),
])
def test_fork_dry_run_finds_source_files_under_the_parent_id(base_dir, monkeypatch, capsys, output_file_text, original_text):
    """The fork's ID does not exist yet during a dry-run, so source files must be looked up under the parent's."""
    with sqlite3.connect(base_dir / "conversations" / f"{CONV}.db") as conn:
        conn.execute("UPDATE steps SET step_payload = ? WHERE idx = 4", (original_text.encode(),))
    if output_file_text:
        output_dir = base_dir / "brain" / CONV / ".system_generated" / "steps" / "4"
        output_dir.mkdir(parents=True)
        (output_dir / "output.txt").write_text(output_file_text, encoding="utf-8")
    before = snapshot(base_dir)
    run_cli(monkeypatch, base_dir, "3", "--replace-text", NEW_TEXT, "--fork", "b", "--dry-run")
    assert f"[DRY-RUN] String replacement: '{original_text}' -> '{NEW_TEXT}'" in capsys.readouterr().out.splitlines()
    assert snapshot(base_dir) == before


# ---------------------------------------------------------------------------
# A replacement that cannot be applied stops with an error and changes nothing
# ---------------------------------------------------------------------------

def not_found(text: str, step: int) -> str:
    return f"Could not find '{text}' in step {step}, so the replacement was not applied. Nothing was changed."


def empty_replacement(step: int) -> str:
    return f"The replacement text is empty, so step {step} would not change. Nothing was changed."


ABSENT = ["--replace-old", "NOT THERE", "--replace-new", NEW_TEXT]
FILE_READ_RESULT = tool_result_payload("ORIGINAL FILE TEXT")


@pytest.mark.parametrize("step, options, message", [
    ("1", ABSENT, not_found("NOT THERE", 2)),
    ("3", ABSENT, not_found("NOT THERE", 4)),
    ("5", ABSENT, not_found("NOT THERE", 6)),
    ("7", ABSENT, not_found("NOT THERE", 8)),
    ("9", ABSENT, not_found("NOT THERE", 10)),
    ("1", ["--replace-old", "ORIGINAL", "--replace-new", ""], empty_replacement(2)),
    ("1", ["--replace-old", "ORIGINAL"], empty_replacement(2)),
    ("11", ABSENT, not_found("NOT THERE", 12)),
    ("17", ABSENT, not_found("NOT THERE", 17)),
    ("21", ABSENT, not_found("NOT THERE", 21)),
])
@pytest.mark.parametrize("mode", [[], ["--dry-run"], ["--fork", "b"], ["--fork", "b", "--dry-run"]])
def test_replacement_that_cannot_apply_stops_and_changes_nothing(base_dir, monkeypatch, capsys, step, options, message, mode):
    """These used to delete the later steps and report success, in most cases without any warning."""
    before = originals(base_dir)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, step, *options, *mode)
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == error_box(message)
    assert_no_fork_exists(base_dir)
    assert originals(base_dir) == before


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(text=st.text(min_size=1, max_size=100).filter(
    lambda t: t.strip() and t.encode() not in FILE_READ_RESULT and t.strip().encode() not in FILE_READ_RESULT
))
def test_any_absent_text_is_refused_without_changes(base_dir, text):
    """Invariant: whatever text is searched for, if it is absent then nothing is created or changed."""
    before = originals(base_dir)
    with pytest.raises(ValueError) as error:
        execute_fork(str(base_dir), CONV, 1, branch_name="b", category="tool_call_read",
                     intervention_kwargs={"tool_name": "view_file", "replace_old": text, "replace_new": NEW_TEXT})
    assert str(error.value).endswith("so the replacement was not applied. Nothing was changed.")
    assert_no_fork_exists(base_dir)
    assert originals(base_dir) == before


def test_replacement_on_a_task_tool_call_still_applies_when_the_text_exists(base_dir, monkeypatch):
    run_cli(monkeypatch, base_dir, "11", "--replace-old", "running", "--replace-new", "stopped")
    assert step_payload(base_dir, CONV, 12) == tool_result_payload("task stopped")
    assert step_indices(base_dir, CONV) == list(range(13))


def test_replacement_on_a_notification_still_applies_when_the_text_exists(base_dir, monkeypatch):
    run_cli(monkeypatch, base_dir, "17", "--replace-old", "exit code 0", "--replace-new", "exit code 1")
    payload = step_payload(base_dir, CONV, 17)
    assert b"exit code 1" in payload
    assert b"exit code 0" not in payload
    assert step_indices(base_dir, CONV) == list(range(18))


def test_task_step_with_no_replacement_requested_is_not_treated_as_a_failure(base_dir, monkeypatch):
    """A log option alone asks for no text change, so finding no text to change is not an error."""
    add_task_logs(base_dir)
    with sqlite3.connect(base_dir / "conversations" / f"{CONV}.db") as conn:
        conn.execute("UPDATE steps SET step_payload = ? WHERE idx = 12", (b"opaque",))
    transcript = base_dir / "brain" / CONV / ".system_generated" / "logs" / "transcript_full.jsonl"
    records = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record["step_index"] == 12:
            record["content"] = ""
    transcript.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    run_cli(monkeypatch, base_dir, "11", "--keep-task-log")
    assert step_payload(base_dir, CONV, 12) == b"opaque"
    assert step_indices(base_dir, CONV) == list(range(13))


@pytest.mark.parametrize("answers, message", [
    (("NOT THERE", NEW_TEXT), not_found("NOT THERE", 2)),
    (("ORIGINAL", ""), empty_replacement(2)),
])
def test_intervene_key_reports_a_replacement_that_cannot_apply(base_dir, monkeypatch, answers, message):
    before = snapshot(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, 1, answers=answers)
    app.handle_input_char(ord("i"))
    assert app.status_msg == f"Intervention failed: {message}"
    assert snapshot(base_dir) == before


def test_tui_fork_key_reports_a_replacement_that_cannot_apply(base_dir, monkeypatch):
    """When output.txt differs from the stored result, the fork used to report success with the result unchanged."""
    output_dir = base_dir / "brain" / CONV / ".system_generated" / "steps" / "2"
    output_dir.mkdir(parents=True)
    (output_dir / "output.txt").write_text("TEXT ONLY IN OUTPUT FILE", encoding="utf-8")
    before = originals(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, 1, answers=(NEW_TEXT, "b"))
    app.handle_input_char(ord("f"))
    assert app.status_msg == f"Fork failed: {not_found('TEXT ONLY IN OUTPUT FILE', 2)}"
    assert_no_fork_exists(base_dir)
    assert originals(base_dir) == before


# ---------------------------------------------------------------------------
# A task log is only removed when a simulated error cancels that task's launch
# ---------------------------------------------------------------------------

LAUNCH_TEXT = f"Tool is running as a background task with task id: {CONV}/task-2"
MENTION_TEXT = "Notes: see task-2 for details"
BOTH_LOGS = {"task-2": LOG_2, "task-4": LOG_4}


def transcript_paths(base: Path) -> list[Path]:
    logs = base / "brain" / CONV / ".system_generated" / "logs"
    return [logs / "transcript_full.jsonl", logs / "transcript.jsonl"]


def set_tool_result(base: Path, step: int, text: str) -> None:
    with sqlite3.connect(base / "conversations" / f"{CONV}.db") as conn:
        conn.execute("UPDATE steps SET step_payload = ? WHERE idx = ?", (tool_result_payload(text), step))
    for path in transcript_paths(base):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in records:
            if record["step_index"] == step:
                record["content"] = text
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def remove_finish_records(base: Path) -> None:
    """Without a later finish on record, the logs do not count as future output, so no log option applies."""
    for path in transcript_paths(base):
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(line for line in lines if "finished with result" not in line), encoding="utf-8")


@pytest.mark.parametrize("step, expected", [
    # Step 1's cut point is before task-4 started, so only its log counts as legitimately gone
    (1, {"task-2": LOG_2}),
    (3, BOTH_LOGS),
    (5, BOTH_LOGS),
    (7, BOTH_LOGS),
    (9, BOTH_LOGS),
])
@pytest.mark.parametrize("route", ["command line", "command line fork", "I key", "F key"])
def test_result_that_only_mentions_a_task_keeps_its_log(base_dir, monkeypatch, step, expected, route):
    """Any result containing text like task-2 used to delete that log, even an ordinary file read."""
    remove_finish_records(base_dir)
    add_task_logs(base_dir)
    set_tool_result(base_dir, step + 1, MENTION_TEXT)

    replace = ["--replace-old", "Notes", "--replace-new", NEW_TEXT]
    if route == "command line":
        run_cli(monkeypatch, base_dir, str(step), *replace)
    elif route == "command line fork":
        run_cli(monkeypatch, base_dir, str(step), *replace, "--fork", "b")
    elif route == "I key":
        app, _, _ = open_tui(monkeypatch, base_dir, step, answers=("Notes", NEW_TEXT))
        app.handle_input_char(ord("i"))
    else:
        app, _, _ = open_tui(monkeypatch, base_dir, step, answers=(NEW_TEXT, "b"))
        app.handle_input_char(ord("f"))

    conv_id = only_fork(base_dir)[0] if "fork" in route or route == "F key" else CONV
    assert NEW_TEXT.encode() in step_payload(base_dir, conv_id, step + 1)
    assert task_logs(base_dir, conv_id) == expected


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(before=st.text(max_size=50), after=st.text(max_size=50))
def test_any_result_text_mentioning_a_task_keeps_its_log(base_dir, before, after):
    """Invariant: no text around a task's name makes a replacement delete that task's log."""
    remove_finish_records(base_dir)
    add_task_logs(base_dir)
    set_tool_result(base_dir, 8, f"{before}task-2{after}")
    fork_id = execute_fork(str(base_dir), CONV, 7, branch_name="b", category="tool_call_web",
                           intervention_kwargs={"tool_name": "read_url_content", "replace_text": NEW_TEXT})
    assert task_logs(base_dir, fork_id) == BOTH_LOGS


@pytest.mark.parametrize("fork_options", [[], ["--fork", "b"]])
def test_keep_task_log_is_honoured_on_the_command_that_started_the_task(base_dir, monkeypatch, fork_options):
    add_task_logs(base_dir)
    set_tool_result(base_dir, 4, LAUNCH_TEXT)
    run_cli(monkeypatch, base_dir, "3", "--replace-old", "background task", "--replace-new", NEW_TEXT,
            "--keep-task-log", *fork_options)
    conv_id = only_fork(base_dir)[0] if fork_options else CONV
    assert task_logs(base_dir, conv_id) == BOTH_LOGS


@pytest.mark.parametrize("fork_options", [[], ["--fork", "b"]])
def test_simulated_error_on_a_background_launch_still_removes_its_log(base_dir, monkeypatch, capsys, fork_options):
    """This is the one case the cleanup exists for: the task no longer ran, so its log must go."""
    add_task_logs(base_dir)
    set_tool_result(base_dir, 4, LAUNCH_TEXT)
    run_cli(monkeypatch, base_dir, "3", "--simulate-error", "1", *fork_options)
    conv_id = only_fork(base_dir)[0] if fork_options else CONV
    assert "  - Cleaned background task artifacts for 'task-2' (converted to synchronous result)." in capsys.readouterr().out.splitlines()
    assert task_logs(base_dir, conv_id) == {"task-4": LOG_4}


def test_simulated_web_error_keeps_a_log_the_page_only_mentions(base_dir, monkeypatch):
    """A failed page fetch cancels no background task, whatever the page said."""
    add_task_logs(base_dir)
    set_tool_result(base_dir, 8, LAUNCH_TEXT)
    run_cli(monkeypatch, base_dir, "7", "--simulate-error", "404", "--keep-task-log")
    assert task_logs(base_dir, CONV) == BOTH_LOGS


def test_simulated_command_error_keeps_a_log_the_output_only_mentions(base_dir, monkeypatch):
    add_task_logs(base_dir)
    set_tool_result(base_dir, 4, MENTION_TEXT)
    run_cli(monkeypatch, base_dir, "3", "--simulate-error", "1")
    assert task_logs(base_dir, CONV) == BOTH_LOGS


@pytest.mark.parametrize("result, options, previews_removal", [
    (LAUNCH_TEXT, ["--simulate-error", "1"], True),
    (LAUNCH_TEXT, ["--replace-old", "background task", "--replace-new", NEW_TEXT, "--keep-task-log"], False),
    (MENTION_TEXT, ["--simulate-error", "1"], False),
])
def test_dry_run_previews_log_removal_only_for_a_cancelled_launch(base_dir, monkeypatch, capsys, result, options, previews_removal):
    add_task_logs(base_dir)
    set_tool_result(base_dir, 4, result)
    before = snapshot(base_dir)
    run_cli(monkeypatch, base_dir, "3", *options, "--dry-run")
    lines = capsys.readouterr().out.splitlines()
    assert ("[DRY-RUN] Would remove background task log: tasks/task-2.log" in lines) == previews_removal
    assert snapshot(base_dir) == before


# ---------------------------------------------------------------------------
# TUI fork key: which steps it refuses
# ---------------------------------------------------------------------------

def test_fixture_covers_every_expected_step_type(base_dir):
    """Guards the tests below: a fixture drift could otherwise make them pass without testing the right steps."""
    app = RewindTUIApp(str(base_dir), CONV)
    expected = dict(TOOL_CALL_STEPS + NOTIFICATION_STEPS + PROMPT_STEPS + REPLY_STEPS)
    expected.update({step: category for step, category, _ in REFUSED_STEPS})
    assert {step.idx: step.category for step in app.loader.steps if step.idx in expected} == expected


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("step, category, message", REFUSED_STEPS)
def test_tui_fork_action_refuses_unsupported_step(base_dir, monkeypatch, step, category, message, dry_run):
    before = snapshot(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, step, dry_run=dry_run)
    assert app.execute_action_fork("b", replace_text=NEW_TEXT) == (False, message, None)
    assert app.status_msg == message
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("step, category, message", REFUSED_STEPS)
def test_tui_fork_key_refuses_before_asking_anything(base_dir, monkeypatch, step, category, message):
    before = snapshot(base_dir)
    app, inputs, confirms = open_tui(monkeypatch, base_dir, step)
    app.handle_input_char(ord("f"))
    assert (inputs, confirms) == ([], [])
    assert app.status_msg == message
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("cursor", [-1, 22, 2 ** 63])
def test_tui_fork_with_out_of_range_cursor_does_nothing(base_dir, monkeypatch, cursor):
    before = snapshot(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, cursor)
    assert app.execute_action_fork("b") == (False, "No step selected.", None)
    assert snapshot(base_dir) == before


# ---------------------------------------------------------------------------
# TUI fork key: prompts and agent replies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step, category", PROMPT_STEPS)
def test_tui_fork_key_on_prompt_cuts_there_and_stays_idle(base_dir, monkeypatch, step, category):
    before = originals(base_dir)
    app, inputs, _ = open_tui(monkeypatch, base_dir, step)
    app.handle_input_char(ord("f"))
    fork_id, parent_id, fork_step, branch, label = only_fork(base_dir)
    assert (parent_id, fork_step, branch, label) == (CONV, step, "b", "")
    assert inputs == [branch_question(step)]
    assert app.status_msg == f"Forked to branch 'b' ({fork_id[:8]}...)."
    assert step_indices(base_dir, fork_id) == list(range(step))
    assert run_state(base_dir, fork_id) == ("CASCADE_RUN_STATUS_IDLE", 0)
    assert originals(base_dir, fork_id) == before


@pytest.mark.parametrize("step, category", REPLY_STEPS)
def test_tui_fork_key_on_reply_sets_the_fork_to_continue(base_dir, monkeypatch, step, category):
    """The fork key used to skip this, unlike the command line, while still labelling the fork as regenerating."""
    before = originals(base_dir)
    app, inputs, _ = open_tui(monkeypatch, base_dir, step)
    app.handle_input_char(ord("f"))
    fork_id, parent_id, fork_step, branch, label = only_fork(base_dir)
    assert (parent_id, fork_step, branch, label) == (CONV, step, "b", "before_response_regeneration")
    assert inputs == [branch_question(step)]
    assert step_indices(base_dir, fork_id) == list(range(step))
    assert run_state(base_dir, fork_id) == ("CASCADE_RUN_STATUS_RUNNING", 1)
    assert originals(base_dir, fork_id) == before


# ---------------------------------------------------------------------------
# TUI fork key: replacement text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step, category", TOOL_CALL_STEPS)
def test_tui_fork_key_replaces_the_whole_tool_result(base_dir, monkeypatch, step, category):
    before = originals(base_dir)
    app, inputs, _ = open_tui(monkeypatch, base_dir, step, answers=(NEW_TEXT, "b"))
    app.handle_input_char(ord("f"))
    fork_id, parent_id, fork_step, branch, label = only_fork(base_dir)
    assert (parent_id, fork_step, branch, label) == (CONV, step, "b", "tool_content_replacement")
    assert inputs == ["Replacement text for the whole tool result:", branch_question(step)]
    assert app.status_msg == f"Forked to branch 'b' ({fork_id[:8]}...)."
    assert step_indices(base_dir, fork_id) == list(range(step + 2))
    assert step_payload(base_dir, fork_id, step + 1) == tool_result_payload(NEW_TEXT)
    assert run_state(base_dir, fork_id) == ("CASCADE_RUN_STATUS_RUNNING", 1)
    assert originals(base_dir, fork_id) == before


@pytest.mark.parametrize("step, category", NOTIFICATION_STEPS)
def test_tui_fork_key_changes_only_the_selected_notification(base_dir, monkeypatch, step, category):
    """The fork key used to leave the selected notification alone and blank the step after it."""
    before = originals(base_dir)
    app, inputs, _ = open_tui(monkeypatch, base_dir, step, answers=(NEW_TEXT, "b"))
    app.handle_input_char(ord("f"))
    fork_id, parent_id, fork_step, branch, label = only_fork(base_dir)
    assert (parent_id, fork_step, branch, label) == (CONV, step, "b", "tool_content_replacement")
    assert inputs == ["Replacement text for the whole message:", branch_question(step)]
    assert step_indices(base_dir, fork_id) == list(range(step + 1))
    assert NEW_TEXT.encode() in step_payload(base_dir, fork_id, step)
    assert [i for i in range(step) if step_payload(base_dir, fork_id, i) != step_payload(base_dir, CONV, i)] == []
    assert run_state(base_dir, fork_id) == ("CASCADE_RUN_STATUS_RUNNING", 1)
    assert originals(base_dir, fork_id) == before


@pytest.mark.parametrize("step", [1, 11, 17])
@pytest.mark.parametrize("text", [None, "", "   "])
def test_tui_fork_key_cancels_without_replacement_text(base_dir, monkeypatch, step, text):
    before = snapshot(base_dir)
    app, inputs, _ = open_tui(monkeypatch, base_dir, step, answers=(text,))
    app.handle_input_char(ord("f"))
    assert len(inputs) == 1
    assert app.status_msg == "Fork cancelled: no replacement text given."
    assert snapshot(base_dir) == before


def test_tui_fork_key_cancels_when_branch_name_is_escaped(base_dir, monkeypatch):
    before = snapshot(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, 1, answers=(NEW_TEXT, None))
    app.handle_input_char(ord("f"))
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("step, category", TOOL_CALL_STEPS + NOTIFICATION_STEPS + PROMPT_STEPS + REPLY_STEPS)
def test_tui_fork_dry_run_creates_nothing(base_dir, monkeypatch, step, category):
    before = snapshot(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, step, dry_run=True)
    success, message, fork_id = app.execute_action_fork("b", replace_text=NEW_TEXT)
    assert (success, message) == (True, f"Forked to branch 'b' ({fork_id[:8]}...) [DRY-RUN].")
    assert snapshot(base_dir) == before


# ---------------------------------------------------------------------------
# TUI fork key: the task log question
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step, answers", [(16, ("b",)), (3, (NEW_TEXT, "b")), (11, (NEW_TEXT, "b"))])
def test_tui_fork_clears_future_logs_on_yes(base_dir, monkeypatch, step, answers):
    add_task_logs(base_dir)
    before = originals(base_dir)
    app, _, confirms = open_tui(monkeypatch, base_dir, step, answers=answers, clear_logs=True)
    app.handle_input_char(ord("f"))
    fork_id = only_fork(base_dir)[0]
    assert confirms[0] == log_question(2)
    assert task_logs(base_dir, fork_id) == {"task-2": "", "task-4": ""}
    assert originals(base_dir, fork_id) == before


def test_tui_fork_keeps_replacement_text_when_clearing_logs_on_a_task_tool_call(base_dir, monkeypatch):
    """Clearing logs must not cost the replacement, which the task engine used to drop."""
    add_task_logs(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, 11, answers=(NEW_TEXT, "b"), clear_logs=True)
    app.handle_input_char(ord("f"))
    assert step_payload(base_dir, only_fork(base_dir)[0], 12) == tool_result_payload(NEW_TEXT)


@pytest.mark.parametrize("step, answers", [(16, ("b",)), (3, (NEW_TEXT, "b")), (11, (NEW_TEXT, "b"))])
def test_tui_fork_creates_nothing_on_no(base_dir, monkeypatch, step, answers):
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    app, _, confirms = open_tui(monkeypatch, base_dir, step, answers=answers, clear_logs=False)
    app.handle_input_char(ord("f"))
    assert confirms == [log_question(2)]
    assert app.status_msg == logs_kept_message(2)
    assert snapshot(base_dir) == before


def test_tui_fork_at_tool_call_counts_the_task_its_own_result_started(base_dir, monkeypatch):
    """A tool call keeps its result step, so task-2, named after that step, is still future output."""
    add_task_logs(base_dir)
    app, _, confirms = open_tui(monkeypatch, base_dir, 1, answers=(NEW_TEXT, "b"), clear_logs=True)
    app.handle_input_char(ord("f"))
    assert confirms[0] == log_question(1)
    assert task_logs(base_dir, only_fork(base_dir)[0]) == {"task-2": ""}


def test_tui_fork_on_notification_clears_only_future_logs(base_dir, monkeypatch):
    """task-4 finished at the selected step, so its log is not future output and must survive."""
    add_task_logs(base_dir)
    app, _, confirms = open_tui(monkeypatch, base_dir, 17, answers=(NEW_TEXT, "b"), clear_logs=True)
    app.handle_input_char(ord("f"))
    assert confirms[0] == log_question(1)
    assert task_logs(base_dir, only_fork(base_dir)[0]) == {"task-2": "", "task-4": LOG_4}


def test_tui_fork_asks_nothing_about_logs_when_none_hold_future_output(base_dir, monkeypatch):
    add_task_logs(base_dir)
    app, _, confirms = open_tui(monkeypatch, base_dir, 21, answers=(NEW_TEXT, "b"))
    app.handle_input_char(ord("f"))
    fork_id = only_fork(base_dir)[0]
    assert confirms == [f"Forked to {fork_id[:8]}... Switch TUI to new branch? [y/N]:"]
    assert task_logs(base_dir, fork_id) == {"task-2": LOG_2, "task-4": LOG_4}


def test_tui_fork_dry_run_writes_nothing_after_yes(base_dir, monkeypatch):
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    app, _, confirms = open_tui(monkeypatch, base_dir, 16, dry_run=True, clear_logs=True)
    app.handle_input_char(ord("f"))
    assert confirms == [log_question(2)]
    assert snapshot(base_dir) == before


def test_tui_fork_action_refuses_undecided_logs(base_dir, monkeypatch):
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, 16)
    assert app.execute_action_fork("b") == (False, f"Fork failed: {UNDECIDED_LOGS}", None)
    assert snapshot(base_dir) == before


# ---------------------------------------------------------------------------
# TUI rewind and intervene keys: the task log question
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["r", "b"])
def test_rewind_keys_clear_future_logs_on_yes(base_dir, monkeypatch, key):
    add_task_logs(base_dir)
    app, _, confirms = open_tui(monkeypatch, base_dir, 16, clear_logs=True)
    app.handle_input_char(ord(key))
    assert confirms[1] == log_question(2)
    assert step_indices(base_dir, CONV) == list(range(16))
    assert task_logs(base_dir, CONV) == {"task-2": "", "task-4": ""}


@pytest.mark.parametrize("key", ["r", "b"])
def test_rewind_keys_change_nothing_on_no(base_dir, monkeypatch, key):
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    app, _, confirms = open_tui(monkeypatch, base_dir, 16)
    app.handle_input_char(ord(key))
    assert confirms[1] == log_question(2)
    assert app.status_msg == logs_kept_message(2)
    assert snapshot(base_dir) == before


def test_rewind_key_asks_nothing_about_logs_when_none_hold_future_output(base_dir, monkeypatch):
    app, _, confirms = open_tui(monkeypatch, base_dir, 16)
    app.handle_input_char(ord("r"))
    assert confirms == ["Rewind to Step 16? Deletes 6 downstream step(s). [y/N]: "]
    assert step_indices(base_dir, CONV) == list(range(16))


def test_rewind_key_dry_run_writes_nothing_after_yes(base_dir, monkeypatch):
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, 16, dry_run=True, clear_logs=True)
    app.handle_input_char(ord("r"))
    assert snapshot(base_dir) == before


def test_intervene_key_clears_future_logs_on_yes(base_dir, monkeypatch):
    add_task_logs(base_dir)
    app, _, confirms = open_tui(monkeypatch, base_dir, 3, answers=("command output", NEW_TEXT), clear_logs=True)
    app.handle_input_char(ord("i"))
    assert confirms == [log_question(2)]
    assert step_payload(base_dir, CONV, 4) == tool_result_payload(NEW_TEXT)
    assert task_logs(base_dir, CONV) == {"task-2": "", "task-4": ""}


def test_intervene_key_changes_nothing_on_no(base_dir, monkeypatch):
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    app, _, confirms = open_tui(monkeypatch, base_dir, 3, answers=("command output", NEW_TEXT))
    app.handle_input_char(ord("i"))
    assert confirms == [log_question(2)]
    assert app.status_msg == logs_kept_message(2)
    assert snapshot(base_dir) == before


def test_tui_actions_refuse_undecided_logs_without_the_dialogs(base_dir, monkeypatch):
    """The checks sit in the actions as well as the keys, so no route through the TUI can skip them."""
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, 16)
    assert app.execute_action_rewind() == (False, logs_kept_message(2))
    app.cursor = 3
    assert app.execute_action_intervene(replace_old="command output", replace_new=NEW_TEXT) == (False, logs_kept_message(2))
    assert snapshot(base_dir) == before
