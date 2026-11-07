"""
Tests for conversation forks: which forks are refused, what a successful fork
produces, and that the original session is never changed.

Also covers the TUI's question about task logs, because rewinding to an earlier
step leaves those logs holding output from steps that no longer exist.

Refusal tests compare every file in the session byte for byte, so any write is
caught, including writes to the child subagent conversation.
"""

import contextlib
import fcntl
import gc
import json
import os
import pty
import resource
import select
import shutil
import socket
import sqlite3
import struct
import sys
import termios
import time
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from rewinding.Antigravity_Rewind import execute_fork, main
from rewinding.tui import RewindTUIApp

CONV = "11111111-1111-1111-1111-111111111111"
CHILD = "22222222-2222-2222-2222-222222222222"
TRAJECTORY = "33333333-3333-3333-3333-333333333333"

LOG_2 = "TASK 2 FINAL OUTPUT\n"
LOG_4 = "TASK 4 FINAL OUTPUT\n"
NEW_TEXT = "NEW TEXT"

UNDECIDED_LOGS = "Task logs hold output from steps this fork discards: task-4, task-2. Decide what they should contain."

# Steps the TUI fork key must refuse, as (step index, detected category, status message).
REFUSED_STEPS = [
    (2, "tool_result", "Cannot fork at a tool result; select the tool call at step 1."),
    (13, "tool_call_other", "Cannot fork at step 13: results of invoke_subagent cannot be changed."),
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


@contextlib.contextmanager
def open_db(path: Path):
    """Commits on success and always closes, which a plain "with sqlite3.connect" does not do."""
    conn = sqlite3.connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


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
    conn.execute("CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT, trajectory_type INTEGER, source INTEGER)")
    conn.execute("INSERT INTO trajectory_meta VALUES (?, ?, 1, 1)", (TRAJECTORY, path.stem))
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
    with open_db(base / "conversations" / f"{conv_id}.db") as conn:
        return [row[0] for row in conn.execute("SELECT idx FROM steps ORDER BY idx")]


def step_payload(base: Path, conv_id: str, idx: int) -> bytes:
    with open_db(base / "conversations" / f"{conv_id}.db") as conn:
        return conn.execute("SELECT step_payload FROM steps WHERE idx = ?", (idx,)).fetchone()[0]


def catalog_row(base: Path, conv_id: str) -> tuple:
    with open_db(base / "conversation_summaries.db") as conn:
        return conn.execute("SELECT * FROM conversation_summaries WHERE conversation_id = ?", (conv_id,)).fetchone()


def run_state(base: Path, conv_id: str) -> tuple[str, int]:
    with open_db(base / "conversation_summaries.db") as conn:
        return conn.execute(
            "SELECT status, not_fully_idle FROM conversation_summaries WHERE conversation_id = ?", (conv_id,)
        ).fetchone()


def lineage(base: Path) -> list[tuple]:
    with open_db(base / "conversation_summaries.db") as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'conversation_lineage'").fetchone():
            return []
        return conn.execute(
            "SELECT fork_id, parent_id, fork_step, branch_name, intervention_type FROM conversation_lineage"
        ).fetchall()


def task_logs(base: Path, conv_id: str) -> dict[str, str]:
    folder = base / "brain" / conv_id / ".system_generated" / "tasks"
    return {path.stem: path.read_text(encoding="utf-8") for path in folder.glob("*.log")}


def descendants(base: Path, conv_id: str) -> list[str]:
    with open_db(base / "conversation_summaries.db") as conn:
        children = [row[0] for row in conn.execute(
            "SELECT conversation_id FROM conversation_summaries WHERE parent_conversation_id = ?", (conv_id,))]
    return [found for child in children for found in [child] + descendants(base, child)]


def originals(base: Path, fork_id: str | None = None) -> dict:
    """
    The state a successful fork must leave alone: every file except the catalog and
    the fork's own files, plus the parent's and child's rows in the catalog.
    """
    files = snapshot(base)
    del files["conversation_summaries.db"]
    if fork_id:
        for conv_id in [fork_id] + descendants(base, fork_id):
            files = {
                name: data for name, data in files.items()
                if name != f"conversations/{conv_id}.db" and not name.startswith(f"brain/{conv_id}/")
            }
    return {"files": files, "parent_row": catalog_row(base, CONV), "child_row": catalog_row(base, CHILD)}


def assert_no_fork_exists(base: Path) -> None:
    assert sorted(path.stem for path in (base / "conversations").iterdir()) == [CONV, CHILD]
    assert sorted(path.name for path in (base / "brain").iterdir()) == [CONV, CHILD]
    with open_db(base / "conversation_summaries.db") as conn:
        rows = conn.execute("SELECT conversation_id FROM conversation_summaries ORDER BY conversation_id").fetchall()
    assert rows == [(CONV,), (CHILD,)]
    assert lineage(base) == []


def run_cli(monkeypatch, base: Path, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["agy-rewind", "-d", str(base), "-c", CONV, *args])
    main()


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
    app.text_dialogs = []
    monkeypatch.setattr(app, "_show_text_dialog", lambda title, lines: app.text_dialogs.append((title, lines)), raising=False)
    return app, inputs, confirms


def only_fork(base: Path) -> tuple:
    [row] = lineage(base)
    return row


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
    with open_db(base_dir / "conversations" / f"{CONV}.db") as conn:
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
    with open_db(base_dir / "conversations" / f"{CONV}.db") as conn:
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
    with open_db(base_dir / "conversations" / f"{CONV}.db") as conn:
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
    with open_db(base_dir / "conversations" / f"{CONV}.db") as conn:
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


def test_tui_fork_key_replaces_a_result_whose_output_file_differs(base_dir, monkeypatch):
    """A stale output.txt once left the stored result unchanged, while the fork reported success."""
    output_dir = base_dir / "brain" / CONV / ".system_generated" / "steps" / "2"
    output_dir.mkdir(parents=True)
    (output_dir / "output.txt").write_text("TEXT ONLY IN OUTPUT FILE", encoding="utf-8")
    app, _, _ = open_tui(monkeypatch, base_dir, 1, answers=(NEW_TEXT, "b"))
    app.handle_input_char(ord("f"))
    fork_id = only_fork(base_dir)[0]
    assert app.status_msg == f"Forked to branch 'b' ({fork_id[:8]}...)."
    assert step_payload(base_dir, fork_id, 2) == tool_result_payload(NEW_TEXT)


def test_tui_fork_key_reports_a_replacement_that_cannot_apply(base_dir, monkeypatch):
    """A result storing no text at all, as some real ones do, must fail rather than report success."""
    with open_db(base_dir / "conversations" / f"{CONV}.db") as conn:
        conn.execute("UPDATE steps SET step_payload = ? WHERE idx = 2", (int_field(1, 132) + int_field(4, 3),))
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
    with open_db(base / "conversations" / f"{CONV}.db") as conn:
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
    run_cli(monkeypatch, base_dir, "3", "--simulate-error", "1", "--keep-task-log", *fork_options)
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
    run_cli(monkeypatch, base_dir, "3", "--simulate-error", "1", "--keep-task-log")
    assert task_logs(base_dir, CONV) == BOTH_LOGS


@pytest.mark.parametrize("result, options, previews_removal", [
    (LAUNCH_TEXT, ["--simulate-error", "1", "--keep-task-log"], True),
    (LAUNCH_TEXT, ["--replace-old", "background task", "--replace-new", NEW_TEXT, "--keep-task-log"], False),
    (MENTION_TEXT, ["--simulate-error", "1", "--keep-task-log"], False),
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
# A log option on its own changes the log, not what the agent was told
# ---------------------------------------------------------------------------

def transcript_content(base: Path, conv_id: str, step: int, name: str = "transcript_full") -> str:
    logs = base / "brain" / conv_id / ".system_generated" / "logs"
    paths = {"transcript_full": logs / "transcript_full.jsonl", "chunk": logs / "chunks" / "transcript_full" / "00000000.jsonl"}
    records = [json.loads(line) for line in paths[name].read_text(encoding="utf-8").splitlines()]
    return next(record["content"] for record in records if record["step_index"] == step)


@pytest.mark.parametrize("target, result_step, options, expected_logs", [
    (11, 12, ["--keep-task-log"], BOTH_LOGS),
    (11, 12, ["--task-output", ""], {"task-2": "", "task-4": ""}),
    (11, 12, ["--task-output", "partial"], {"task-2": "partial", "task-4": "partial"}),
    (17, 17, ["--keep-task-log"], BOTH_LOGS),
    (17, 17, ["--task-output", ""], {"task-2": "", "task-4": ""}),
    (17, 17, ["--task-output", "partial"], {"task-2": "partial", "task-4": "partial"}),
    (21, 21, ["--keep-task-log"], BOTH_LOGS),
    (21, 21, ["--task-output", ""], {"task-2": "", "task-4": LOG_4}),
])
@pytest.mark.parametrize("fork_options", [[], ["--fork", "b"]])
def test_log_option_alone_leaves_the_result_unchanged(base_dir, monkeypatch, target, result_step, options, expected_logs, fork_options):
    """A log option used to blank a manage_task result and rewrite a notification as a clean exit."""
    add_task_logs(base_dir)
    payload_before = step_payload(base_dir, CONV, result_step)
    transcript_before = transcript_content(base_dir, CONV, result_step)

    run_cli(monkeypatch, base_dir, str(target), *options, *fork_options)

    conv_id = only_fork(base_dir)[0] if fork_options else CONV
    # A fork's data names the fork, so that is the one expected difference
    assert step_payload(base_dir, conv_id, result_step) == payload_before.replace(CONV.encode(), conv_id.encode())
    assert transcript_content(base_dir, conv_id, result_step) == transcript_before.replace(CONV, conv_id)
    assert transcript_content(base_dir, conv_id, result_step, "chunk") == transcript_before.replace(CONV, conv_id)
    assert step_indices(base_dir, conv_id) == list(range(result_step + 1))
    assert task_logs(base_dir, conv_id) == expected_logs


def test_log_option_alone_keeps_a_failed_task_reported_as_failed(base_dir, monkeypatch):
    """Rewriting the message as a clean exit would flip what the agent was told about the task."""
    failed = system_message_payload("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", f"{CONV}/task-4",
                                    f'Task id "{CONV}/task-4" finished with result:\n\nThe command exited with code 7.')
    with open_db(base_dir / "conversations" / f"{CONV}.db") as conn:
        conn.execute("UPDATE steps SET step_payload = ? WHERE idx = 17", (failed,))
    add_task_logs(base_dir)
    run_cli(monkeypatch, base_dir, "17", "--keep-task-log")
    assert step_payload(base_dir, CONV, 17) == failed


def test_status_option_still_changes_the_result(base_dir, monkeypatch):
    """Guards the other direction: only a log option on its own should leave the result alone."""
    run_cli(monkeypatch, base_dir, "11", "--task-status", "failed", "--task-exit-code", "3")
    assert step_payload(base_dir, CONV, 12) == tool_result_payload('Task task status: failed (exit code 3)')


# ---------------------------------------------------------------------------
# The I key uses the right code for each step type
# ---------------------------------------------------------------------------

I_REFUSED_STEPS = [
    (2, "Step 2 (tool_result) cannot be intervened upon."),
    (0, "Step 0 (prompt) cannot be intervened upon."),
    (16, "Step 16 (model_response) cannot be intervened upon."),
    (18, "Step 18 (system_message) cannot be intervened upon."),
    (13, "Cannot intervene on step 13: results of invoke_subagent cannot be changed."),
    (15, "Cannot change a subagent report here; use --subagent-exit on the command line."),
]


def retype_step(base: Path, step: int, step_type: int, payload: bytes) -> None:
    with open_db(base / "conversations" / f"{CONV}.db") as conn:
        conn.execute("UPDATE steps SET step_type = ?, step_payload = ? WHERE idx = ?", (step_type, payload, step))


@pytest.mark.parametrize("step, message", I_REFUSED_STEPS)
def test_intervene_key_refuses_before_asking_anything(base_dir, monkeypatch, step, message):
    before = snapshot(base_dir)
    app, inputs, confirms = open_tui(monkeypatch, base_dir, step)
    app.handle_input_char(ord("i"))
    assert (inputs, confirms) == ([], [])
    assert app.status_msg == message
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("step, message", I_REFUSED_STEPS)
def test_intervene_action_refuses_the_same_steps(base_dir, monkeypatch, step, message):
    before = snapshot(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, step)
    assert app.execute_action_intervene(replace_old="a", replace_new="b") == (False, message)
    assert snapshot(base_dir) == before


def test_intervene_key_refuses_a_report_even_when_a_tool_result_follows(base_dir, monkeypatch):
    """I used to change the tool result after the report instead, and report success."""
    retype_step(base_dir, 16, 132, tool_result_payload("finished in the NEXT step"))
    before = snapshot(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, 15, answers=("finished", "CHANGED"))
    app.handle_input_char(ord("i"))
    assert app.status_msg == "Cannot change a subagent report here; use --subagent-exit on the command line."
    assert snapshot(base_dir) == before


def test_intervene_key_changes_the_selected_notification_even_when_a_tool_result_follows(base_dir, monkeypatch):
    """I used to change the tool result after the notification instead, and report success."""
    retype_step(base_dir, 18, 132, tool_result_payload("exit code 0 in the NEXT step"))
    app, _, _ = open_tui(monkeypatch, base_dir, 17, answers=("exit code 0", "exit code 1"))
    app.handle_input_char(ord("i"))
    assert b"exit code 1" in step_payload(base_dir, CONV, 17)
    assert step_indices(base_dir, CONV) == list(range(18))


def test_intervene_key_on_manage_task_replaces_part_of_the_result(base_dir, monkeypatch):
    app, _, _ = open_tui(monkeypatch, base_dir, 11, answers=("running", "stopped"))
    app.handle_input_char(ord("i"))
    assert app.status_msg == "Intervened on step 11 successfully."
    assert step_payload(base_dir, CONV, 12) == tool_result_payload("task stopped")
    assert step_indices(base_dir, CONV) == list(range(13))
    assert run_state(base_dir, CONV) == ("CASCADE_RUN_STATUS_RUNNING", 1)


def test_intervene_key_on_manage_task_keeps_a_log_that_is_not_future_output(base_dir, monkeypatch):
    """I used to delete this log because the result named the task."""
    remove_finish_records(base_dir)
    add_task_logs(base_dir)
    set_tool_result(base_dir, 12, 'Task "task-2" status: RUNNING')
    app, _, confirms = open_tui(monkeypatch, base_dir, 11, answers=("RUNNING", "FAILED"))
    app.handle_input_char(ord("i"))
    assert confirms == []
    assert step_payload(base_dir, CONV, 12) == tool_result_payload('Task "task-2" status: FAILED')
    assert task_logs(base_dir, CONV) == BOTH_LOGS


def test_intervene_key_on_notification_changes_that_message(base_dir, monkeypatch):
    app, _, _ = open_tui(monkeypatch, base_dir, 17, answers=("exit code 0", "exit code 1"))
    app.handle_input_char(ord("i"))
    assert app.status_msg == "Intervened on step 17 successfully."
    payload = step_payload(base_dir, CONV, 17)
    assert b"exit code 1" in payload
    assert b"exit code 0" not in payload
    assert step_indices(base_dir, CONV) == list(range(18))
    assert run_state(base_dir, CONV) == ("CASCADE_RUN_STATUS_RUNNING", 1)


def test_intervene_key_on_notification_clears_only_future_logs(base_dir, monkeypatch):
    add_task_logs(base_dir)
    app, _, confirms = open_tui(monkeypatch, base_dir, 17, answers=("exit code 0", "exit code 1"), clear_logs=True)
    app.handle_input_char(ord("i"))
    assert confirms == [log_question(1)]
    assert task_logs(base_dir, CONV) == {"task-2": "", "task-4": LOG_4}


@pytest.mark.parametrize("step, check_step", [(11, 12), (17, 17)])
def test_intervene_key_with_no_search_text_replaces_the_whole_task_text(base_dir, monkeypatch, step, check_step):
    """A blank search text means replace everything, as it already does on ordinary tool calls."""
    app, _, _ = open_tui(monkeypatch, base_dir, step, answers=("", NEW_TEXT))
    app.handle_input_char(ord("i"))
    assert NEW_TEXT.encode() in step_payload(base_dir, CONV, check_step)
    assert step_indices(base_dir, CONV) == list(range(check_step + 1))


@pytest.mark.parametrize("step, answers, message", [
    (1, ("", ""), "Intervene mode on step 1 ('view_file') requires intervention content."),
    (11, ("", ""), "The replacement text is empty, so step 12 would not change. Nothing was changed."),
    (11, ("running", ""), "The replacement text is empty, so step 12 would not change. Nothing was changed."),
    (17, ("exit code 0", ""), "The replacement text is empty, so step 17 would not change. Nothing was changed."),
    (11, ("NOT THERE", NEW_TEXT), "Could not find 'NOT THERE' in step 12, so the replacement was not applied. Nothing was changed."),
    (17, ("NOT THERE", NEW_TEXT), "Could not find 'NOT THERE' in step 17, so the replacement was not applied. Nothing was changed."),
])
def test_intervene_key_reports_failures_on_one_line_and_changes_nothing(base_dir, monkeypatch, step, answers, message):
    before = snapshot(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, step, answers=answers)
    app.handle_input_char(ord("i"))
    assert app.status_msg == f"Intervention failed: {message}"
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("step, answers", [(11, ("running", "stopped")), (17, ("exit code 0", "exit code 1"))])
def test_intervene_key_dry_run_writes_nothing_on_task_steps(base_dir, monkeypatch, step, answers):
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, step, dry_run=True, answers=answers, clear_logs=True)
    app.handle_input_char(ord("i"))
    assert app.status_msg == f"Intervened on step {step} successfully [DRY-RUN]."
    assert snapshot(base_dir) == before


# ---------------------------------------------------------------------------
# Helper reports accept the task log options that the log check asks for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("log_options, expected", [
    (["--task-output", ""], {"task-2": "", "task-4": ""}),
    (["--task-output", "task-2=x", "--task-output", "task-4=y"], {"task-2": "x", "task-4": "y"}),
    (["--keep-task-log"], BOTH_LOGS),
])
def test_subagent_report_accepts_log_options_when_a_task_finishes_later(base_dir, monkeypatch, log_options, expected):
    """The log check demanded these options and the report check then rejected them, so this was a dead end."""
    add_task_logs(base_dir)
    run_cli(monkeypatch, base_dir, "15", "--subagent-exit", "crash", *log_options)
    assert b"[FATAL ERROR]" in step_payload(base_dir, CONV, 15)
    assert step_indices(base_dir, CONV) == list(range(17))
    assert task_logs(base_dir, CONV) == expected


def test_subagent_report_is_still_refused_until_logs_are_decided(base_dir, monkeypatch, capsys):
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, "15", "--subagent-exit", "crash")
    assert exit_info.value.code == 1
    assert "Rewind cuts off downstream completion of multiple background tasks" in capsys.readouterr().err
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("options", [["--task-status", "failed"], ["--task-exit-code", "1"], ["--task-exit-code", "0"]])
def test_subagent_report_still_rejects_task_status_options(base_dir, monkeypatch, capsys, options):
    before = snapshot(base_dir)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, "15", "--subagent-exit", "crash", *options)
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == (
        f"Error: --task-status and --task-exit-code are only supported for background task steps (got '{CHILD}').\n"
    )
    assert snapshot(base_dir) == before


def test_out_dir_alternative_also_works_when_a_task_finishes_later(base_dir, tmp_path, monkeypatch):
    """The fork refusal recommends --out-dir, so that route must not hit the dead end either."""
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    copy_dir = tmp_path / "copy"
    run_cli(monkeypatch, base_dir, "15", "--subagent-exit", "crash", "--task-output", "", "--out-dir", str(copy_dir))
    assert snapshot(base_dir) == before
    assert task_logs(copy_dir, CONV) == {"task-2": "", "task-4": ""}
    assert step_indices(copy_dir, CHILD) == [0, 1, 2]


# ---------------------------------------------------------------------------
# A simulated command error still asks about other tasks' logs
# ---------------------------------------------------------------------------

def test_simulated_command_error_is_refused_until_logs_are_decided(base_dir, monkeypatch, capsys):
    """This route used to skip the log check, leaving future output in other tasks' logs."""
    add_task_logs(base_dir)
    before = snapshot(base_dir)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, "3", "--simulate-error", "1")
    assert exit_info.value.code == 1
    assert "Rewind cuts off downstream completion of multiple background tasks" in capsys.readouterr().err
    assert snapshot(base_dir) == before


def test_simulated_error_on_a_launch_asks_only_about_other_tasks(base_dir, monkeypatch, capsys):
    """The launch's own task never ran after this, so its log is removed and needs no decision."""
    add_task_logs(base_dir)
    set_tool_result(base_dir, 4, LAUNCH_TEXT)
    before = snapshot(base_dir)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, "3", "--simulate-error", "1")
    assert exit_info.value.code == 1
    assert "Error: Rewind cuts off downstream completion of background task 'task-4' (completed at Step 17)." in capsys.readouterr().err
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("log_options, expected", [
    (["--task-output", ""], {"task-4": ""}),
    (["--task-output", "task-4=partial"], {"task-4": "partial"}),
    (["--keep-task-log"], {"task-4": LOG_4}),
])
@pytest.mark.parametrize("fork_options", [[], ["--fork", "b"]])
def test_simulated_error_on_a_launch_applies_log_choices_to_other_tasks(base_dir, monkeypatch, log_options, expected, fork_options):
    add_task_logs(base_dir)
    set_tool_result(base_dir, 4, LAUNCH_TEXT)
    run_cli(monkeypatch, base_dir, "3", "--simulate-error", "1", *log_options, *fork_options)
    conv_id = only_fork(base_dir)[0] if fork_options else CONV
    assert task_logs(base_dir, conv_id) == expected


def test_simulated_error_on_a_launch_needs_no_log_option_when_no_other_task_is_affected(base_dir, monkeypatch):
    add_task_logs(base_dir)
    (base_dir / "brain" / CONV / ".system_generated" / "tasks" / "task-4.log").unlink()
    set_tool_result(base_dir, 4, LAUNCH_TEXT)
    run_cli(monkeypatch, base_dir, "3", "--simulate-error", "1")
    assert task_logs(base_dir, CONV) == {}


def test_execute_fork_refuses_undecided_logs_for_a_simulated_command_error(base_dir):
    add_task_logs(base_dir)
    set_tool_result(base_dir, 4, LAUNCH_TEXT)
    before = originals(base_dir)
    with pytest.raises(ValueError) as error:
        execute_fork(str(base_dir), CONV, 3, branch_name="b", category="tool_call_command",
                     intervention_kwargs={"tool_name": "run_command", "simulate_error": "1"})
    assert str(error.value) == "Task logs hold output from steps this fork discards: task-4. Decide what they should contain."
    assert_no_fork_exists(base_dir)
    assert originals(base_dir) == before


# ---------------------------------------------------------------------------
# Per-turn metadata shrinks along with the model steps
# ---------------------------------------------------------------------------

def assert_metadata_matches_model_steps(base: Path, conv_id: str) -> None:
    """gen_metadata and executor_metadata hold one row per model turn, numbered from zero."""
    with open_db(base / "conversations" / f"{conv_id}.db") as conn:
        turns = conn.execute("SELECT COUNT(*) FROM steps WHERE step_type = 15").fetchone()[0]
        gen = [row[0] for row in conn.execute("SELECT idx FROM gen_metadata ORDER BY idx")]
        executor = [row[0] for row in conn.execute("SELECT idx FROM executor_metadata ORDER BY idx")]
    assert (gen, executor) == (list(range(turns)), list(range(turns)))


NOT_REPORT_STEPS = [
    ("16", []),
    ("19", []),
    ("1", ["--replace-text", NEW_TEXT]),
    ("11", ["--replace-text", NEW_TEXT]),
    ("11", ["--task-status", "failed"]),
    ("11", ["--keep-task-log"]),
    ("17", ["--replace-text", NEW_TEXT]),
    ("17", ["--task-status", "failed"]),
    ("21", ["--keep-task-log"]),
]


@pytest.mark.parametrize("step, options", NOT_REPORT_STEPS + [("15", ["--subagent-exit", "crash"])])
def test_in_place_changes_trim_metadata_to_the_remaining_model_steps(base_dir, monkeypatch, step, options):
    run_cli(monkeypatch, base_dir, step, *options)
    assert_metadata_matches_model_steps(base_dir, CONV)


@pytest.mark.parametrize("step, options", NOT_REPORT_STEPS)
def test_forks_trim_metadata_to_the_remaining_model_steps(base_dir, monkeypatch, step, options):
    run_cli(monkeypatch, base_dir, step, *options, "--fork", "b")
    assert_metadata_matches_model_steps(base_dir, only_fork(base_dir)[0])
    assert_metadata_matches_model_steps(base_dir, CONV)


# ---------------------------------------------------------------------------
# A replaced task result stays consistent in every record
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("options, expected", [
    (["--replace-old", "running", "--replace-new", "stopped"], "task stopped"),
    (["--replace-text", NEW_TEXT], NEW_TEXT),
    (["--keep-task-log"], "task running"),
])
@pytest.mark.parametrize("fork_options", [[], ["--fork", "b"]])
def test_task_tool_call_keeps_every_record_of_the_result_in_step(base_dir, monkeypatch, options, expected, fork_options):
    """The database and transcript were updated but output.txt kept the old text, so the records disagreed."""
    output_dir = base_dir / "brain" / CONV / ".system_generated" / "steps" / "12"
    output_dir.mkdir(parents=True)
    (output_dir / "output.txt").write_text("task running", encoding="utf-8")

    run_cli(monkeypatch, base_dir, "11", *options, *fork_options)

    conv_id = only_fork(base_dir)[0] if fork_options else CONV
    output_file = base_dir / "brain" / conv_id / ".system_generated" / "steps" / "12" / "output.txt"
    assert step_payload(base_dir, conv_id, 12) == tool_result_payload(expected)
    assert transcript_content(base_dir, conv_id, 12) == expected
    assert output_file.read_text(encoding="utf-8") == expected


@pytest.mark.parametrize("options", [["--replace-text", NEW_TEXT], ["--keep-task-log"]])
def test_notification_keeps_the_output_folder_of_its_own_step(base_dir, monkeypatch, options):
    """The selected step survives the cut, so its folder must too, and it must not gain a generated output.txt."""
    output_dir = base_dir / "brain" / CONV / ".system_generated" / "steps"
    (output_dir / "17").mkdir(parents=True)
    (output_dir / "17" / "note.txt").write_text("kept", encoding="utf-8")
    (output_dir / "18").mkdir()
    run_cli(monkeypatch, base_dir, "17", *options)
    assert sorted(path.name for path in output_dir.iterdir()) == ["17"]
    assert sorted(path.name for path in (output_dir / "17").iterdir()) == ["note.txt"]


# ---------------------------------------------------------------------------
# The TUI never prints over its own screen
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key, step, answers", [
    ("r", 16, ()),
    ("b", 16, ()),
    ("i", 1, ("ORIGINAL", NEW_TEXT)),
    ("i", 11, ("running", "stopped")),
    ("i", 17, ("exit code 0", "exit code 1")),
    ("f", 1, (NEW_TEXT, "b")),
    ("f", 11, (NEW_TEXT, "b")),
    ("f", 16, ("b",)),
    ("f", 17, (NEW_TEXT, "b")),
    ("i", 1, ("NOT THERE", NEW_TEXT)),
])
@pytest.mark.parametrize("dry_run", [False, True])
def test_tui_actions_print_nothing_to_the_terminal(base_dir, monkeypatch, capfd, key, step, answers, dry_run):
    """Printed messages used to overwrite the screen, and stayed there after the next keypress."""
    add_task_logs(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, step, dry_run=dry_run, answers=answers, clear_logs=True)
    app.handle_input_char(ord(key))
    assert capfd.readouterr() == ("", "")


def test_dry_run_preview_is_shown_in_a_dialog(base_dir, monkeypatch):
    """Capturing the output must not lose the preview, which is the whole point of a dry-run."""
    app, _, _ = open_tui(monkeypatch, base_dir, 16, dry_run=True)
    app.handle_input_char(ord("f"))
    [(title, lines)] = app.text_dialogs
    assert title == "Dry-run preview"
    assert f"[DRY-RUN] Fork source: {CONV}" in lines
    assert "[DRY-RUN] Steps remaining: 16" in lines


@pytest.mark.parametrize("key, answers", [("r", ()), ("b", ())])
def test_rewind_key_dry_run_preview_is_shown_in_a_dialog(base_dir, monkeypatch, key, answers):
    app, _, _ = open_tui(monkeypatch, base_dir, 16, dry_run=True, answers=answers)
    app.handle_input_char(ord(key))
    [(title, lines)] = app.text_dialogs
    assert title == "Dry-run preview"
    assert "[DRY-RUN] Steps remaining: 16" in lines


def test_intervene_key_dry_run_preview_is_shown_in_a_dialog(base_dir, monkeypatch):
    app, _, _ = open_tui(monkeypatch, base_dir, 1, dry_run=True, answers=("ORIGINAL", NEW_TEXT))
    app.handle_input_char(ord("i"))
    [(title, lines)] = app.text_dialogs
    assert title == "Dry-run preview"
    assert "[DRY-RUN] Steps remaining: 3 (steps 0 to 2)" in lines


def test_notice_from_the_tool_is_shown_in_a_dialog(base_dir, monkeypatch):
    """The task-finished message changes while its log does not, which the tool flags and the user must see."""
    add_task_logs(base_dir)
    app, _, _ = open_tui(monkeypatch, base_dir, 17, answers=(NEW_TEXT, "b"), clear_logs=True)
    app.handle_input_char(ord("f"))
    [(title, lines)] = app.text_dialogs
    assert title == "Notice"
    assert lines[0] == "Error: Step 17 task notification was modified, but 'tasks/task-4.log' (20 bytes) was not updated."
    assert not any(set(line.strip()) == {"!"} for line in lines)


@pytest.mark.parametrize("key, step, answers", [
    ("r", 16, ()),
    ("f", 1, (NEW_TEXT, "b")),
    ("i", 1, ("ORIGINAL", NEW_TEXT)),
    ("i", 1, ("NOT THERE", NEW_TEXT)),
])
def test_no_dialog_appears_when_there_is_nothing_to_show(base_dir, monkeypatch, key, step, answers):
    """A normal success or a failure already shows in the status bar."""
    app, _, _ = open_tui(monkeypatch, base_dir, step, answers=answers)
    app.handle_input_char(ord(key))
    assert app.text_dialogs == []


# ---------------------------------------------------------------------------
# The real TUI in a real terminal, because curses crashes cannot show up otherwise
# ---------------------------------------------------------------------------

ESCAPE, ENTER = "\x1b", "\r"
TUI_SCRIPT = main.__code__.co_filename


def run_real_tui(base: Path, rows: int, cols: int, keys: list[str]) -> tuple[int, str]:
    """Runs the TUI in a pseudo-terminal of the given size, presses the keys, and returns its exit code and output."""
    pid, fd = pty.fork()
    if pid == 0:
        fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        os.environ.update(TERM="xterm", ESCDELAY="25", PYTHONDONTWRITEBYTECODE="1")
        # curses trusts these over the terminal, which would silently test the wrong size
        os.environ.pop("COLUMNS", None)
        os.environ.pop("LINES", None)
        os.execv(sys.executable, [sys.executable, TUI_SCRIPT, "-d", str(base), "-c", CONV, "--tui"])

    output = bytearray()
    poller = select.poll()
    poller.register(fd, select.POLLIN | select.POLLHUP)

    def read_until_quiet(quiet: float) -> None:
        while poller.poll(quiet * 1000):
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return
            if not chunk:
                return
            output.extend(chunk)

    poller.poll(10_000)
    read_until_quiet(0.2)
    for key in keys + ["q"]:
        try:
            os.write(fd, key.encode())
        except OSError:
            break
        read_until_quiet(0.08)

    deadline = time.time() + 5
    while time.time() < deadline:
        finished, status = os.waitpid(pid, os.WNOHANG)
        if finished:
            break
        read_until_quiet(0.05)
    else:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        raise AssertionError("The TUI did not exit when asked to quit.")
    os.close(fd)
    return os.waitstatus_to_exitcode(status), output.decode("utf-8", "replace")


@pytest.mark.slow
@pytest.mark.parametrize("rows, cols", [(12, 60), (24, 66), (24, 80), (50, 200)])
@pytest.mark.parametrize("keys", [
    ["j", "j", "k"],
    ["d", "j", "d"],
    ["j", "f", ESCAPE],
    ["j", "i", ESCAPE],
    ["d", "j", "f", "text", ENTER, "b", ENTER, " ", " ", " "],
    ["j", "j", "j", "f", "text", ENTER, "b", ENTER, "n"],
    ["r", "y", "j", "k"],
], ids=["scroll", "dry-run toggle", "fork dialog", "intervene dialog", "dry-run fork popup", "real fork", "rewind to empty"])
def test_real_tui_survives_every_supported_window_size(base_dir, rows, cols, keys):
    exit_code, output = run_real_tui(base_dir, rows, cols, keys)
    assert "Traceback" not in output
    assert exit_code == 0


@pytest.mark.parametrize("width", [60, 66, 67, 79, 80, 81, 120, 300])
@pytest.mark.parametrize("dry_run", [False, True])
def test_header_fits_the_window_and_keeps_the_dry_run_tag_visible(base_dir, monkeypatch, width, dry_run):
    """Dry-run must stay visible on a narrow window, so losing the tag to truncation would be misleading."""
    app, _, _ = open_tui(monkeypatch, base_dir, 0, dry_run=dry_run)
    line = app.header_line(width)
    assert len(line) == width
    assert line.startswith(" [DRY-RUN ON]") == dry_run
    assert "AGY_REWIND" in line


# ---------------------------------------------------------------------------
# Database connections are closed, so large sessions cannot exhaust file handles
# ---------------------------------------------------------------------------

def open_database_handles() -> int:
    links = [os.path.join("/proc/self/fd", name) for name in os.listdir("/proc/self/fd")]
    targets = []
    for link in links:
        try:
            targets.append(os.readlink(link))
        except OSError:
            pass
    return sum(1 for target in targets if ".db" in target)


@pytest.mark.parametrize("action", [
    lambda base, monkeypatch: RewindTUIApp(str(base), CONV),
    lambda base, monkeypatch: run_cli(monkeypatch, base, "19"),
    lambda base, monkeypatch: run_cli(monkeypatch, base, "1", "--replace-text", NEW_TEXT),
    lambda base, monkeypatch: run_cli(monkeypatch, base, "11", "--replace-text", NEW_TEXT),
    lambda base, monkeypatch: run_cli(monkeypatch, base, "19", "--fork", "b"),
    lambda base, monkeypatch: run_cli(monkeypatch, base, "--tree"),
    lambda base, monkeypatch: run_cli(monkeypatch, base, "--list-branches"),
    lambda base, monkeypatch: run_cli(monkeypatch, base, "--show-step", "1"),
    lambda base, monkeypatch: run_cli(monkeypatch, base, "--list"),
    lambda base, monkeypatch: run_cli(monkeypatch, base, "--cleanup"),
], ids=["open tui", "rewind", "tool change", "task change", "fork", "tree", "branches", "show step", "list", "cleanup"])
def test_actions_leave_no_database_handles_open(base_dir, monkeypatch, action):
    """Garbage collection is off, so only an explicit close can pass this."""
    gc.collect()
    gc.disable()
    try:
        before = open_database_handles()
        action(base_dir, monkeypatch)
        assert open_database_handles() == before
    finally:
        gc.enable()


def test_large_session_is_classified_correctly_within_a_low_file_limit(tmp_path):
    """Running out of handles made lookups fail, which the TUI silently showed as an unknown step type."""
    base = tmp_path / "antigravity-cli"
    (base / "conversations").mkdir(parents=True)
    create_conversation_db(base / "conversations" / f"{CONV}.db",
                           [(idx, 14 if idx % 2 == 0 else 15, b"text") for idx in range(300)])
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    gc.collect()
    gc.disable()
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
    try:
        steps = RewindTUIApp(str(base), CONV).loader.steps
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
        gc.enable()
    assert [step.category for step in steps] == ["prompt" if idx % 2 == 0 else "model_response" for idx in range(300)]


# ---------------------------------------------------------------------------
# Forks include recent steps that are still in SQLite's write-ahead log
# ---------------------------------------------------------------------------

def add_steps_that_stay_in_the_log(base: Path) -> sqlite3.Connection:
    """Leaves steps 22 and 23 only in the -wal file, as during a running session or after a crash."""
    conn = sqlite3.connect(base / "conversations" / f"{CONV}.db")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("INSERT INTO steps VALUES (22, 14, 3, ?)", (b"recent prompt",))
    conn.execute("INSERT INTO steps VALUES (23, 14, 3, ?)", (b"recent prompt two",))
    conn.commit()
    return conn


def test_fork_includes_steps_still_in_the_write_ahead_log(base_dir):
    """Antigravity's databases use WAL mode, so copying only the .db file would lose these steps."""
    writer = add_steps_that_stay_in_the_log(base_dir)
    try:
        fork_id = execute_fork(str(base_dir), CONV, 23, branch_name="b", category="prompt")
        assert step_indices(base_dir, fork_id) == list(range(23))
        assert step_payload(base_dir, fork_id, 22) == b"recent prompt"
    finally:
        writer.close()


def test_forked_database_is_intact_and_keeps_its_journal_mode(base_dir):
    writer = add_steps_that_stay_in_the_log(base_dir)
    try:
        fork_id = execute_fork(str(base_dir), CONV, 23, branch_name="b", category="prompt")
    finally:
        writer.close()
    conversations = base_dir / "conversations"
    assert sorted(path.name for path in conversations.iterdir() if path.name.startswith(fork_id)) == [f"{fork_id}.db"]
    with open_db(conversations / f"{fork_id}.db") as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def test_forking_a_wal_database_at_rest_leaves_the_parent_file_unchanged(base_dir):
    """Opening the parent must not change its bytes or leave side files beside it."""
    with open_db(base_dir / "conversations" / f"{CONV}.db") as conn:
        conn.execute("PRAGMA journal_mode=WAL")
    parent = base_dir / "conversations" / f"{CONV}.db"
    before = parent.read_bytes()
    fork_id = execute_fork(str(base_dir), CONV, 19, branch_name="b", category="prompt")
    assert parent.read_bytes() == before
    names = {path.name for path in (base_dir / "conversations").iterdir()}
    copies = {f"{conv_id}.db" for conv_id in descendants(base_dir, fork_id)}
    assert names == {f"{CONV}.db", f"{CHILD}.db", f"{fork_id}.db"} | copies
    with open_db(base_dir / "conversations" / f"{fork_id}.db") as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)


@pytest.mark.parametrize("folder_name", ["with space", "with?question", "with#hash", "with%percent", "한글"])
def test_fork_copies_a_database_whose_path_has_unusual_characters(tmp_path, folder_name):
    """Opening a file read-only goes through a URI, where these characters have special meanings."""
    base = tmp_path / folder_name / "antigravity-cli"
    (base / "conversations").mkdir(parents=True)
    create_conversation_db(base / "conversations" / f"{CONV}.db", [(0, 14, b"prompt"), (1, 15, b"reply"), (2, 14, b"prompt")])
    fork_id = execute_fork(str(base), CONV, 2, branch_name="b", category="prompt")
    assert step_indices(base, fork_id) == [0, 1]


# ---------------------------------------------------------------------------
# Results shaped like real Antigravity data
# ---------------------------------------------------------------------------

def real_result_payload(text: str, combined_output: str = "combined") -> bytes:
    """
    The shape real tool results have: the text at 140.2.1, beside key/value pairs, and a command's
    own record of its output at 140.2.6, inside an Any message holding field 28.
    """
    command = bytes_field(28, bytes_field(21, bytes_field(1, combined_output.encode())))
    any_message = bytes_field(1, b"type.googleapis.com/gemini_coder.Step") + bytes_field(2, command)
    observation = bytes_field(1, text.encode()) + bytes_field(6, any_message)
    pair = bytes_field(1, b"key") + bytes_field(2, b"value")
    return int_field(1, 132) + int_field(4, 3) + bytes_field(140, bytes_field(1, pair) + bytes_field(2, observation))


def set_real_result(base: Path, step: int, text: str, with_output_file: bool = True) -> None:
    """Stores a real-shaped result. Real transcripts add a timestamp header, and output.txt holds the exact bytes."""
    with open_db(base / "conversations" / f"{CONV}.db") as conn:
        conn.execute("UPDATE steps SET step_payload = ? WHERE idx = ?", (real_result_payload(text), step))
    for path in transcript_paths(base):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in records:
            if record["step_index"] == step:
                record["content"] = f"Created At: 2026-10-19T05:10:00Z\nCompleted At: 2026-10-19T05:10:01Z\n\n{text}"
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    if with_output_file:
        folder = base / "brain" / CONV / ".system_generated" / "steps" / str(step)
        folder.mkdir(parents=True)
        (folder / "output.txt").write_bytes(text.encode())


CRLF_OUTPUT = "line one\r\nline two\r\n"
# Each tool call step with the result step after it
TOOL_CALLS_AND_RESULTS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]


@pytest.mark.parametrize("call, result", TOOL_CALLS_AND_RESULTS)
@pytest.mark.parametrize("mode", [[], ["--fork", "b"], ["--dry-run"]])
def test_absent_search_text_on_a_real_shaped_result_stops_and_changes_nothing(base_dir, monkeypatch, capsys, call, result, mode):
    """On real results the whole text used to be replaced when the search text was missing."""
    set_real_result(base_dir, result, "actual text")
    before = originals(base_dir)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, str(call), "--replace-old", "NOT THERE", "--replace-new", NEW_TEXT, *mode)
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == error_box(not_found("NOT THERE", result))
    assert_no_fork_exists(base_dir)
    assert originals(base_dir) == before


@pytest.mark.parametrize("call, result", TOOL_CALLS_AND_RESULTS)
@pytest.mark.parametrize("text", [CRLF_OUTPUT, "plain text", "ends with newline\n", "한글 결과\r\n"])
@pytest.mark.parametrize("fork_options", [[], ["--fork", "b"]])
def test_whole_replacement_applies_to_a_real_shaped_result(base_dir, monkeypatch, call, result, text, fork_options):
    """Reading output.txt in text mode changed CRLF to LF, so the text was never found and nothing changed."""
    set_real_result(base_dir, result, text)
    run_cli(monkeypatch, base_dir, str(call), "--replace-text", NEW_TEXT, *fork_options)
    conv_id = only_fork(base_dir)[0] if fork_options else CONV
    assert step_payload(base_dir, conv_id, result) == real_result_payload(NEW_TEXT, NEW_TEXT)


@pytest.mark.parametrize("call, result", TOOL_CALLS_AND_RESULTS)
def test_partial_replacement_keeps_the_rest_of_a_real_shaped_result_exact(base_dir, monkeypatch, call, result):
    set_real_result(base_dir, result, CRLF_OUTPUT)
    run_cli(monkeypatch, base_dir, str(call), "--replace-old", "line two", "--replace-new", "CHANGED")
    expected = "line one\r\nCHANGED\r\n"
    assert step_payload(base_dir, CONV, result) == real_result_payload(expected)
    output_file = base_dir / "brain" / CONV / ".system_generated" / "steps" / str(result) / "output.txt"
    assert output_file.read_bytes() == expected.encode()
    assert expected in transcript_content(base_dir, CONV, result)


@pytest.mark.parametrize("call, result", TOOL_CALLS_AND_RESULTS)
def test_tui_fork_key_replaces_a_real_shaped_result(base_dir, monkeypatch, call, result):
    set_real_result(base_dir, result, CRLF_OUTPUT)
    app, _, _ = open_tui(monkeypatch, base_dir, call, answers=(NEW_TEXT, "b"))
    app.handle_input_char(ord("f"))
    assert step_payload(base_dir, only_fork(base_dir)[0], result) == real_result_payload(NEW_TEXT, NEW_TEXT)


def test_command_replacement_also_updates_its_recorded_output(base_dir, monkeypatch):
    with open_db(base_dir / "conversations" / f"{CONV}.db") as conn:
        conn.execute("UPDATE steps SET step_payload = ? WHERE idx = 4", (real_result_payload("old output", "old output"),))
    run_cli(monkeypatch, base_dir, "3", "--replace-old", "old", "--replace-new", "new")
    assert step_payload(base_dir, CONV, 4) == real_result_payload("new output", "new output")


def test_simulated_command_error_on_a_real_shaped_result_still_applies(base_dir, monkeypatch, capsys):
    """Guards the exit code route of the same function, which must keep reporting its change."""
    set_real_result(base_dir, 4, CRLF_OUTPUT)
    run_cli(monkeypatch, base_dir, "3", "--simulate-error", "7:denied")
    payload = step_payload(base_dir, CONV, 4)
    assert b"The command exited with code 7." in payload
    assert b"denied" in payload
    assert b"line one" not in payload


# ---------------------------------------------------------------------------
# A step that makes several tool calls keeps a result for each call
# ---------------------------------------------------------------------------

def make_step_call_several_tools(base: Path, step: int, tool: str, results: dict[int, str]) -> None:
    """Real steps can make several calls at once, with one result step per call straight after."""
    with open_db(base / "conversations" / f"{CONV}.db") as conn:
        for result_step, text in results.items():
            conn.execute("UPDATE steps SET step_type = 132, step_payload = ? WHERE idx = ?",
                         (tool_result_payload(text), result_step))
    for path in transcript_paths(base):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in records:
            if record["step_index"] == step:
                record["tool_calls"] = [{"name": tool, "args": {}} for _ in results]
            elif record["step_index"] in results:
                record.pop("tool_calls", None)
                record["type"] = "TOOL_RESULT"
                record["content"] = results[record["step_index"]]
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


THREE_READS = {2: "first file", 3: "second file", 4: "third file"}


@pytest.mark.parametrize("fork_options", [[], ["--fork", "b"]])
def test_changing_one_of_several_calls_keeps_the_other_results(base_dir, monkeypatch, fork_options):
    """Cutting after the first result left the other calls with no result, which the agent cannot continue from."""
    make_step_call_several_tools(base_dir, 1, "view_file", THREE_READS)
    run_cli(monkeypatch, base_dir, "1", "--replace-text", NEW_TEXT, *fork_options)
    conv_id = only_fork(base_dir)[0] if fork_options else CONV
    assert step_indices(base_dir, conv_id) == [0, 1, 2, 3, 4]
    assert step_payload(base_dir, conv_id, 2) == tool_result_payload(NEW_TEXT)
    assert step_payload(base_dir, conv_id, 3) == tool_result_payload("second file")
    assert step_payload(base_dir, conv_id, 4) == tool_result_payload("third file")
    assert transcript_content(base_dir, conv_id, 3) == "second file"
    assert transcript_content(base_dir, conv_id, 4) == "third file"
    assert_metadata_matches_model_steps(base_dir, conv_id)


def test_output_folders_of_the_other_results_are_kept(base_dir, monkeypatch):
    make_step_call_several_tools(base_dir, 1, "view_file", THREE_READS)
    folder = base_dir / "brain" / CONV / ".system_generated" / "steps"
    for step in (3, 4, 5):
        (folder / str(step)).mkdir(parents=True)
        (folder / str(step) / "output.txt").write_text(f"step {step}", encoding="utf-8")
    run_cli(monkeypatch, base_dir, "1", "--replace-text", NEW_TEXT)
    assert sorted(path.name for path in folder.iterdir()) == ["2", "3", "4"]
    assert (folder / "3" / "output.txt").read_text(encoding="utf-8") == "step 3"


@pytest.mark.parametrize("key, answers", [("f", (NEW_TEXT, "b")), ("i", ("first", "changed"))])
def test_tui_keeps_the_other_results_of_several_calls(base_dir, monkeypatch, key, answers):
    make_step_call_several_tools(base_dir, 1, "view_file", THREE_READS)
    app, _, _ = open_tui(monkeypatch, base_dir, 1, answers=answers)
    app.handle_input_char(ord(key))
    conv_id = only_fork(base_dir)[0] if key == "f" else CONV
    assert step_indices(base_dir, conv_id) == [0, 1, 2, 3, 4]
    assert step_payload(base_dir, conv_id, 3) == tool_result_payload("second file")


def test_task_tool_call_keeps_the_other_results_of_several_calls(base_dir, monkeypatch):
    make_step_call_several_tools(base_dir, 11, "manage_task", {12: "first task", 13: "second task"})
    run_cli(monkeypatch, base_dir, "11", "--replace-text", NEW_TEXT)
    assert step_indices(base_dir, CONV) == list(range(14))
    assert step_payload(base_dir, CONV, 12) == tool_result_payload(NEW_TEXT)
    assert step_payload(base_dir, CONV, 13) == tool_result_payload("second task")
    assert transcript_content(base_dir, CONV, 13) == "second task"
    assert_metadata_matches_model_steps(base_dir, CONV)


def test_dry_run_reports_the_other_results_as_remaining(base_dir, monkeypatch, capsys):
    make_step_call_several_tools(base_dir, 1, "view_file", THREE_READS)
    before = snapshot(base_dir)
    run_cli(monkeypatch, base_dir, "1", "--replace-text", NEW_TEXT, "--dry-run")
    assert "[DRY-RUN] Steps remaining: 5 (steps 0 to 4)" in capsys.readouterr().out.splitlines()
    assert snapshot(base_dir) == before


def test_several_calls_with_a_missing_result_keep_only_what_exists(base_dir, monkeypatch):
    """An interrupted session can record more calls than results, and the next step must not be treated as one."""
    make_step_call_several_tools(base_dir, 1, "view_file", THREE_READS)
    with open_db(base_dir / "conversations" / f"{CONV}.db") as conn:
        conn.execute("UPDATE steps SET step_type = 15 WHERE idx = 3")
    run_cli(monkeypatch, base_dir, "1", "--replace-text", NEW_TEXT)
    assert step_indices(base_dir, CONV) == [0, 1, 2]


def test_single_call_is_unchanged_by_the_several_calls_rule(base_dir, monkeypatch):
    """Step 2 is followed by a tool call and then a result, which must not count as a second result."""
    run_cli(monkeypatch, base_dir, "1", "--replace-text", NEW_TEXT)
    assert step_indices(base_dir, CONV) == [0, 1, 2]


@pytest.mark.parametrize("fork_options", [[], ["--fork", "b"]])
def test_task_started_by_a_later_call_in_the_step_counts_as_future_output(base_dir, monkeypatch, capsys, fork_options):
    """task-3 is named after the second result, which survives the cut, so its log is still future output."""
    make_step_call_several_tools(base_dir, 1, "run_command", {2: "started", 3: f"Tool is running as a background task with task id: {CONV}/task-3"})
    tasks = base_dir / "brain" / CONV / ".system_generated" / "tasks"
    tasks.mkdir()
    (tasks / "task-3.log").write_text("FUTURE OUTPUT\n", encoding="utf-8")
    set_tool_result(base_dir, 21, f'Task id "{CONV}/task-3" finished with result:\n\nexit code 0')
    before = snapshot(base_dir)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, "1", "--replace-text", NEW_TEXT, *fork_options)
    assert exit_info.value.code == 1
    assert "Error: Rewind cuts off downstream completion of background task 'task-3' (completed at Step 21)." in capsys.readouterr().err
    assert snapshot(base_dir) == before


def test_real_directory_listing_tool_can_be_changed(base_dir, monkeypatch):
    """Real sessions call this tool list_dir, which was not recognised, so it was refused as an unknown tool."""
    make_step_call_several_tools(base_dir, 1, "list_dir", {2: "file_a file_b"})
    run_cli(monkeypatch, base_dir, "1", "--replace-text", NEW_TEXT)
    assert step_payload(base_dir, CONV, 2) == tool_result_payload(NEW_TEXT)


# ---------------------------------------------------------------------------
# A fork is a conversation of its own, not a copy still claiming to be its parent
# ---------------------------------------------------------------------------

def trajectory_meta(base: Path, conv_id: str) -> list[tuple]:
    with open_db(base / "conversations" / f"{conv_id}.db") as conn:
        return conn.execute("SELECT trajectory_id, cascade_id, trajectory_type, source FROM trajectory_meta").fetchall()


@pytest.mark.parametrize("category, step, content", [
    ("prompt", 19, {}),
    ("model_response", 16, {}),
    ("tool_call_read", 1, {"tool_name": "view_file", "replace_text": NEW_TEXT}),
    ("tool_call_task", 11, {"detail": "manage_task", "replace_text": NEW_TEXT}),
    ("task_notification", 17, {"detail": f"{CONV}/task-4", "replace_text": NEW_TEXT}),
])
def test_forked_database_names_the_fork_as_its_conversation(base_dir, category, step, content):
    """The real agent refused to resume forks ("trajectory not found") because the copy still named its parent."""
    before = originals(base_dir)
    fork_id = execute_fork(str(base_dir), CONV, step, branch_name="b", category=category, intervention_kwargs=content)
    [(trajectory_id, cascade_id, trajectory_type, source)] = trajectory_meta(base_dir, fork_id)
    assert cascade_id == fork_id
    assert (trajectory_type, source) == (1, 1)
    assert trajectory_id != TRAJECTORY and len(trajectory_id) == len(TRAJECTORY)
    assert trajectory_meta(base_dir, CONV) == [(TRAJECTORY, CONV, 1, 1)]
    assert originals(base_dir, fork_id) == before


def test_fork_replaces_the_parents_id_throughout_its_own_data(base_dir):
    """Every real step names its conversation, and task names embed it, which pointed the fork at the parent's logs."""
    add_task_logs(base_dir)
    (base_dir / "brain" / CONV / ".system_generated" / "messages").mkdir()
    message = base_dir / "brain" / CONV / ".system_generated" / "messages" / "m.json"
    message.write_text(json.dumps({"sender": f"{CONV}/task-4", "recipient": CONV}), encoding="utf-8")
    os.utime(message, (1, 1))
    set_tool_result(base_dir, 2, f"Tool is running as a background task with task id: {CONV}/task-2")
    before = originals(base_dir)

    fork_id = execute_fork(str(base_dir), CONV, 19, branch_name="b", category="prompt", task_output=[""])

    with open_db(base_dir / "conversations" / f"{fork_id}.db") as conn:
        for table, column in [("steps", "step_payload"), ("trajectory_meta", "cascade_id"), ("trajectory_meta", "trajectory_id")]:
            for (value,) in conn.execute(f"SELECT {column} FROM {table}"):
                data = value if isinstance(value, bytes) else str(value).encode()
                assert CONV.encode() not in data and TRAJECTORY.encode() not in data
    fork_brain = base_dir / "brain" / fork_id
    assert [path.name for path in fork_brain.rglob("*") if path.is_file() and CONV.encode() in path.read_bytes()] == []
    assert step_payload(base_dir, fork_id, 2) == tool_result_payload(f"Tool is running as a background task with task id: {fork_id}/task-2")
    assert json.loads((fork_brain / ".system_generated" / "messages" / "m.json").read_text(encoding="utf-8")) == {
        "sender": f"{fork_id}/task-4", "recipient": fork_id,
    }
    assert originals(base_dir, fork_id) == before


def test_fork_replaces_an_id_of_a_different_length_without_corrupting_the_data(tmp_path):
    """Protobuf stores lengths, so a replacement of a different length must rewrite them."""
    base = tmp_path / "antigravity-cli"
    (base / "conversations").mkdir(parents=True)
    create_conversation_db(base / "conversations" / "short-id.db",
                           [(0, 14, b"prompt"), (1, 132, tool_result_payload("see short-id/task-1")), (2, 14, b"prompt")])
    fork_id = execute_fork(str(base), "short-id", 2, branch_name="b", category="prompt")
    assert step_payload(base, fork_id, 1) == tool_result_payload(f"see {fork_id}/task-1")


def test_fork_is_not_marked_as_a_subagent_of_its_parent(base_dir):
    """Antigravity identifies subagents by this column, and the fork is a conversation in its own right."""
    fork_id = execute_fork(str(base_dir), CONV, 19, branch_name="b", category="prompt")
    with open_db(base_dir / "conversation_summaries.db") as conn:
        row = conn.execute("SELECT parent_conversation_id, nesting_depth FROM conversation_summaries WHERE conversation_id = ?", (fork_id,)).fetchone()
    assert row == ("", 0)
    assert only_fork(base_dir)[:2] == (fork_id, CONV)


def test_tree_still_shows_a_fork_under_its_parent(base_dir, monkeypatch, capsys):
    fork_id = execute_fork(str(base_dir), CONV, 19, branch_name="b", category="prompt")
    capsys.readouterr()
    run_cli(monkeypatch, base_dir, "--tree")
    assert f"[Fork @ Step 19] {fork_id}" in capsys.readouterr().out


def test_fork_of_a_subagent_keeps_the_link_to_that_subagents_own_parent(base_dir):
    """The copied link is the child's own, so it stays, and the fork gets no extra nesting."""
    with open_db(base_dir / "conversations" / f"{CHILD}.db") as conn:
        conn.execute("INSERT INTO steps VALUES (99, 14, 3, ?)", (b"prompt",))
    fork_id = execute_fork(str(base_dir), CHILD, 99, branch_name="b", category="prompt")
    with open_db(base_dir / "conversation_summaries.db") as conn:
        row = conn.execute("SELECT parent_conversation_id, nesting_depth FROM conversation_summaries WHERE conversation_id = ?", (fork_id,)).fetchone()
    assert row == (CONV, 1)


def test_fork_dry_run_does_not_rewrite_anything(base_dir):
    before = snapshot(base_dir)
    execute_fork(str(base_dir), CONV, 19, branch_name="b", category="prompt", dry_run=True)
    assert snapshot(base_dir) == before


# ---------------------------------------------------------------------------
# The printed resume command is one that really reaches the session
# ---------------------------------------------------------------------------

RESUME_ROUTES = [
    (["19"], None),
    (["16"], None),
    (["1", "--replace-text", NEW_TEXT], None),
    (["11", "--replace-text", NEW_TEXT], None),
    (["17", "--replace-text", NEW_TEXT], None),
    (["19", "--fork", "b"], "fork"),
]


@pytest.mark.parametrize("args, kind", RESUME_ROUTES)
def test_printed_command_sets_home_for_a_session_under_another_home(base_dir, tmp_path, monkeypatch, capsys, args, kind):
    """agy ignores JETSKI_APP_DATA_DIR and finds its data under HOME, so the old command reached the wrong folder."""
    monkeypatch.setenv("HOME", str(tmp_path / "someone_else"))
    run_cli(monkeypatch, base_dir, *args)
    captured = capsys.readouterr()
    output = captured.out + captured.err
    conv_id = only_fork(base_dir)[0] if kind == "fork" else CONV
    assert f'HOME="{tmp_path}" agy --conversation {conv_id}' in output
    assert "JETSKI_APP_DATA_DIR" not in output


@pytest.mark.parametrize("args, kind", RESUME_ROUTES)
def test_printed_command_is_plain_for_a_session_under_the_current_home(base_dir, tmp_path, monkeypatch, capsys, args, kind):
    monkeypatch.setenv("HOME", str(tmp_path))
    run_cli(monkeypatch, base_dir, *args)
    captured = capsys.readouterr()
    output = captured.out + captured.err
    conv_id = only_fork(base_dir)[0] if kind == "fork" else CONV
    assert f"  agy --conversation {conv_id}" in output.splitlines()
    assert 'HOME="' not in output


def test_printed_command_explains_the_layout_for_a_copy_agy_cannot_reach(base_dir, tmp_path, monkeypatch, capsys):
    """A copy made with --out-dir has no .gemini folder, so no command can point agy straight at it."""
    run_cli(monkeypatch, base_dir, "19", "--out-dir", str(tmp_path / "copy"))
    output = capsys.readouterr().out
    assert f'  Move this folder to <new home>/.gemini/antigravity-cli, then run: HOME="<new home>" agy --conversation {CONV}' in output.splitlines()
    assert "JETSKI_APP_DATA_DIR" not in output


# ---------------------------------------------------------------------------
# A fork gets its own copy of every subagent it keeps, so the originals stay safe
# ---------------------------------------------------------------------------

def add_subagent_records(base: Path) -> None:
    """A descriptor in the parent, as real sessions have, and child data that names the parent, as real children do."""
    folder = base / "brain" / CONV / ".system_generated" / "subagents"
    folder.mkdir(parents=True)
    (folder / f"{CHILD}.json").write_text(
        json.dumps({"conversationId": CHILD, "spawnStepIndex": 13, "state": "SUBAGENT_STATE_ALIVE"}), encoding="utf-8")
    with open_db(base / "conversations" / f"{CHILD}.db") as conn:
        conn.execute("UPDATE steps SET step_payload = ? WHERE idx = 1", (f"report to {CONV} from {CHILD}".encode(),))


def originals_with_exact_catalog(base: Path) -> dict:
    """Byte-exact for both original conversations, because a fork must not change them at all."""
    conv_files = {name: data for name, data in snapshot(base).items()
                  if name in (f"conversations/{CONV}.db", f"conversations/{CHILD}.db")
                  or name.startswith((f"brain/{CONV}/", f"brain/{CHILD}/"))}
    return {"files": conv_files, "parent_row": catalog_row(base, CONV), "child_row": catalog_row(base, CHILD)}


def only_child_of(base: Path, conv_id: str) -> str:
    [clone] = descendants(base, conv_id)
    return clone


def catalog_status(base: Path, conv_id: str) -> tuple:
    with open_db(base / "conversation_summaries.db") as conn:
        return conn.execute(
            "SELECT status, killed, step_count, parent_conversation_id, nesting_depth FROM conversation_summaries WHERE conversation_id = ?",
            (conv_id,)).fetchone()


@pytest.mark.parametrize("options, status, killed, in_payload", [
    (["--subagent-exit", "crash"], "CASCADE_RUN_STATUS_KILLED", 1, b"[FATAL ERROR]"),
    (["--subagent-exit", "complete"], "CASCADE_RUN_STATUS_IDLE", 1, b"[TASK COMPLETE]"),
    (["--subagent-exit", "keep-alive", "--replace-text", "edited report"], "CASCADE_RUN_STATUS_IDLE", 0, b"edited report"),
])
def test_fork_at_a_report_changes_only_its_own_copy_of_the_subagent(base_dir, monkeypatch, options, status, killed, in_payload):
    """Forking here used to cut the shared child's history and mark it killed, which damaged the original session."""
    add_subagent_records(base_dir)
    before = originals_with_exact_catalog(base_dir)

    run_cli(monkeypatch, base_dir, "15", *options, "--fork", "b")

    fork_id = only_fork(base_dir)[0]
    clone = only_child_of(base_dir, fork_id)
    assert clone != CHILD
    assert originals_with_exact_catalog(base_dir) == before
    assert step_indices(base_dir, clone) == [0, 1, 2]
    assert catalog_status(base_dir, clone) == (status, killed, 3, fork_id, 1)
    assert in_payload in step_payload(base_dir, fork_id, 15)
    assert CHILD.encode() not in step_payload(base_dir, fork_id, 15)
    assert clone.encode() in step_payload(base_dir, fork_id, 15)


def test_copied_subagent_points_to_the_fork_throughout_its_data(base_dir, monkeypatch):
    add_subagent_records(base_dir)
    run_cli(monkeypatch, base_dir, "19", "--fork", "b")
    fork_id = only_fork(base_dir)[0]
    clone = only_child_of(base_dir, fork_id)
    assert step_indices(base_dir, clone) == list(range(6))
    assert step_payload(base_dir, clone, 1) == f"report to {fork_id} from {clone}".encode()
    assert trajectory_meta(base_dir, clone)[0][1] == clone
    assert transcript_for(base_dir, clone, 0) == "child 0"
    descriptors = base_dir / "brain" / fork_id / ".system_generated" / "subagents"
    assert [path.name for path in descriptors.iterdir()] == [f"{clone}.json"]
    assert json.loads((descriptors / f"{clone}.json").read_text(encoding="utf-8"))["conversationId"] == clone


def transcript_for(base: Path, conv_id: str, step: int) -> str:
    return transcript_content(base, conv_id, step)


def test_original_subagent_is_untouched_by_a_fork_that_keeps_it(base_dir, monkeypatch):
    add_subagent_records(base_dir)
    before = originals_with_exact_catalog(base_dir)
    run_cli(monkeypatch, base_dir, "19", "--fork", "b")
    assert originals_with_exact_catalog(base_dir) == before


def test_fork_from_before_the_subagent_existed_has_no_copy_of_it(base_dir, monkeypatch):
    add_subagent_records(base_dir)
    before = originals_with_exact_catalog(base_dir)
    run_cli(monkeypatch, base_dir, "1", "--replace-text", NEW_TEXT, "--fork", "b")
    fork_id = only_fork(base_dir)[0]
    assert descendants(base_dir, fork_id) == []
    assert list((base_dir / "brain" / fork_id / ".system_generated" / "subagents").iterdir()) == []
    assert originals_with_exact_catalog(base_dir) == before


def test_subagents_of_a_subagent_are_copied_too(base_dir, monkeypatch):
    grandchild = "44444444-4444-4444-4444-444444444444"
    create_conversation_db(base_dir / "conversations" / f"{grandchild}.db", [(0, 14, b"grandchild"), (1, 15, CHILD.encode())])
    with open_db(base_dir / "conversation_summaries.db") as conn:
        conn.execute("INSERT INTO conversation_summaries (conversation_id, title, step_count, parent_conversation_id, nesting_depth) "
                     "VALUES (?, 'Grandchild', 2, ?, 2)", (grandchild, CHILD))
    before = (base_dir / "conversations" / f"{grandchild}.db").read_bytes()

    run_cli(monkeypatch, base_dir, "19", "--fork", "b")

    fork_id = only_fork(base_dir)[0]
    [clone, grandchild_clone] = descendants(base_dir, fork_id)
    assert grandchild_clone != grandchild
    assert step_payload(base_dir, grandchild_clone, 1) == clone.encode()
    assert catalog_status(base_dir, grandchild_clone)[3:] == (clone, 2)
    assert (base_dir / "conversations" / f"{grandchild}.db").read_bytes() == before


def test_failed_fork_at_a_report_removes_its_copies(base_dir, monkeypatch, capsys):
    add_subagent_records(base_dir)
    before = originals_with_exact_catalog(base_dir)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, "15", "--subagent-exit", "keep-alive", "--replace-file", "/nonexistent/file", "--fork", "b")
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == error_box("Replacement file not found: /nonexistent/file")
    assert_no_fork_exists(base_dir)
    assert originals_with_exact_catalog(base_dir) == before


@pytest.mark.parametrize("options", [["--subagent-exit", "crash"], ["--subagent-exit", "keep-alive", "--replace-text", "x"]])
def test_fork_at_a_report_dry_run_creates_nothing(base_dir, monkeypatch, options):
    add_subagent_records(base_dir)
    before = snapshot(base_dir)
    run_cli(monkeypatch, base_dir, "15", *options, "--fork", "b", "--dry-run")
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("log_options, expected", [
    (["--task-output", ""], {"task-2": "", "task-4": ""}),
    (["--keep-task-log"], BOTH_LOGS),
])
def test_fork_at_a_report_applies_log_choices_to_the_fork_only(base_dir, monkeypatch, log_options, expected):
    add_task_logs(base_dir)
    before = originals_with_exact_catalog(base_dir)
    run_cli(monkeypatch, base_dir, "15", "--subagent-exit", "crash", *log_options, "--fork", "b")
    assert task_logs(base_dir, only_fork(base_dir)[0]) == expected
    assert originals_with_exact_catalog(base_dir) == before


@pytest.mark.parametrize("damaged_payload", [b"", b"   ", b"\x00", b"\x00\xff\x00", REPORT_PAYLOAD[:20]])
def test_fork_at_a_report_with_damaged_data_never_touches_the_originals(base_dir, monkeypatch, damaged_payload):
    """The transcript still names the sender, so the step is recognised as a report and must stay safe."""
    with open_db(base_dir / "conversations" / f"{CONV}.db") as conn:
        conn.execute("UPDATE steps SET step_payload = ? WHERE idx = 15", (damaged_payload,))
    before = originals_with_exact_catalog(base_dir)
    with contextlib.suppress(SystemExit):
        run_cli(monkeypatch, base_dir, "15", "--subagent-exit", "crash", "--fork", "b")
    assert originals_with_exact_catalog(base_dir) == before


def test_fork_at_a_report_survives_a_malformed_transcript_record(base_dir, monkeypatch):
    add_subagent_records(base_dir)
    transcript = base_dir / "brain" / CONV / ".system_generated" / "logs" / "transcript_full.jsonl"
    transcript.write_text('{"step_index": 15, "content": \n' + transcript.read_text(encoding="utf-8"), encoding="utf-8")
    before = originals_with_exact_catalog(base_dir)
    run_cli(monkeypatch, base_dir, "15", "--subagent-exit", "crash", "--fork", "b")
    assert b"[FATAL ERROR]" in step_payload(base_dir, only_fork(base_dir)[0], 15)
    assert originals_with_exact_catalog(base_dir) == before


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(branch=st.text(max_size=200))
def test_fork_at_a_report_keeps_the_originals_safe_for_any_branch_name(base_dir, branch):
    before = originals_with_exact_catalog(base_dir)
    execute_fork(str(base_dir), CONV, 15, branch_name=branch, category="subagent_report",
                 intervention_kwargs={"sender_uuid": CHILD, "subagent_exit": "crash"})
    assert originals_with_exact_catalog(base_dir) == before


def test_tui_fork_key_at_a_report_replaces_its_text_without_inventing_a_crash(base_dir, monkeypatch):
    """The key used to add a fake crash unasked, so now it changes only what was typed."""
    add_subagent_records(base_dir)
    before = originals_with_exact_catalog(base_dir)
    app, inputs, _ = open_tui(monkeypatch, base_dir, 15, answers=(NEW_TEXT, "b"))
    app.handle_input_char(ord("f"))
    fork_id = only_fork(base_dir)[0]
    assert inputs == ["Replacement text for the whole message:", branch_question(15)]
    assert app.status_msg == f"Forked to branch 'b' ({fork_id[:8]}...)."
    payload = step_payload(base_dir, fork_id, 15)
    assert NEW_TEXT.encode() in payload
    assert b"[FATAL ERROR]" not in payload
    assert catalog_status(base_dir, only_child_of(base_dir, fork_id))[:2] == ("CASCADE_RUN_STATUS_IDLE", 0)
    assert originals_with_exact_catalog(base_dir) == before


def test_replacement_text_that_names_the_parent_is_kept_as_written(base_dir):
    """What the user typed is the experiment, so the ID rewriting must not alter it."""
    text = f"the parent was {CONV}"
    fork_id = execute_fork(str(base_dir), CONV, 1, branch_name="b", category="tool_call_read",
                           intervention_kwargs={"tool_name": "view_file", "replace_text": text})
    assert step_payload(base_dir, fork_id, 2) == tool_result_payload(text)


def test_search_text_that_names_the_parent_still_finds_it(base_dir):
    set_tool_result(base_dir, 2, f"launched by {CONV}")
    fork_id = execute_fork(str(base_dir), CONV, 1, branch_name="b", category="tool_call_read",
                           intervention_kwargs={"tool_name": "view_file", "replace_old": CONV, "replace_new": NEW_TEXT})
    assert step_payload(base_dir, fork_id, 2) == tool_result_payload(f"launched by {NEW_TEXT}")


def test_fork_copies_the_subagent_it_changes_when_the_catalog_has_no_link_to_it(base_dir, monkeypatch):
    """Without the link no copy was made, so the intervention would have edited the original."""
    with open_db(base_dir / "conversation_summaries.db") as conn:
        conn.execute("UPDATE conversation_summaries SET parent_conversation_id = '' WHERE conversation_id = ?", (CHILD,))
    before = originals_with_exact_catalog(base_dir)
    run_cli(monkeypatch, base_dir, "15", "--subagent-exit", "crash", "--fork", "b")
    assert originals_with_exact_catalog(base_dir) == before


def test_old_message_files_survive_the_id_rewrite(base_dir):
    """Rewriting a file updated its modification time, which later cleanup mistook for a recent message."""
    messages = base_dir / "brain" / CONV / ".system_generated" / "messages"
    messages.mkdir()
    old_message = messages / "old.json"
    old_message.write_text(json.dumps({"sender": CONV}), encoding="utf-8")
    os.utime(old_message, (1000, 1000))
    fork_id = execute_fork(str(base_dir), CONV, 19, branch_name="b", category="prompt")
    copied = base_dir / "brain" / fork_id / ".system_generated" / "messages" / "old.json"
    assert json.loads(copied.read_text(encoding="utf-8")) == {"sender": fork_id}
    assert copied.stat().st_mtime == 1000


# ---------------------------------------------------------------------------
# The ACTF parser is optional, so listing and searching work normally without it
# ---------------------------------------------------------------------------

def test_list_works_without_warnings_when_the_optional_parser_is_absent(base_dir, monkeypatch, capsys):
    """The missing cli.py used to print a warning on every use, as if something were wrong."""
    run_cli(monkeypatch, base_dir, "--list")
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "first prompt" in captured.out
    assert "second prompt" in captured.out
    assert "fallback" not in captured.out.lower()


def test_search_works_without_warnings_when_the_optional_parser_is_absent(base_dir, monkeypatch, capsys):
    run_cli(monkeypatch, base_dir, "second prompt")
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Found 1 matching step(s) for query: 'second prompt'" in captured.out


def test_search_reports_when_nothing_matches(base_dir, monkeypatch, capsys):
    run_cli(monkeypatch, base_dir, "text that appears nowhere")
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "No steps found matching query: 'text that appears nowhere'" in captured.out


def test_sync_option_explains_itself_when_the_optional_parser_is_absent(base_dir, monkeypatch, capsys):
    run_cli(monkeypatch, base_dir, "--list", "--sync")
    captured = capsys.readouterr()
    assert captured.err == "[Note] --sync has no effect without the optional ACTF parser (cli.py).\n"
    assert "first prompt" in captured.out


@pytest.mark.parametrize("args", [["--list"], ["first prompt"], []])
def test_listing_and_searching_change_nothing(base_dir, monkeypatch, capsys, args):
    before = snapshot(base_dir)
    run_cli(monkeypatch, base_dir, *args)
    assert snapshot(base_dir) == before


# ---------------------------------------------------------------------------
# The README describes the tool as it is
# ---------------------------------------------------------------------------

README = Path(TUI_SCRIPT).resolve().parents[2] / "README.md"


def test_readme_only_mentions_options_that_exist(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["agy-rewind", "--help"])
    with pytest.raises(SystemExit):
        main()
    help_text = capsys.readouterr().out
    # agy's own options appear in the README too, so only the tool's commands are checked
    commands = [line for line in README.read_text(encoding="utf-8").splitlines() if "Antigravity_Rewind.py" in line]
    flags = {token.split("=")[0] for line in commands for token in line.split() if token.startswith("--")}
    assert sorted(flag for flag in flags if flag not in help_text) == []
    assert len(flags) >= 8


@pytest.mark.parametrize("required", ["--fork", "--tree", "--list-branches", "--cleanup", "--keep-task-log", "--task-output", "agy --conversation"])
def test_readme_documents_the_main_commands(required):
    assert required in README.read_text(encoding="utf-8")


def test_readme_does_not_recommend_the_setting_agy_ignores():
    assert "JETSKI_APP_DATA_DIR=" not in README.read_text(encoding="utf-8")


def test_readme_says_a_resumed_session_needs_a_prompt():
    """Tested against the real agent: the session waits, so the README must not promise that it continues alone."""
    text = README.read_text(encoding="utf-8")
    assert "does not continue by itself" in text
    assert "arming the session so the LLM regenerates" not in text


@pytest.mark.parametrize("key, description", [("F", "Fork"), ("R", "Rewind"), ("B", "before"), ("I", "Intervene"), ("D", "dry-run"), ("Q", "Quit")])
def test_readme_lists_every_tui_key(key, description):
    lines = [line for line in README.read_text(encoding="utf-8").splitlines() if line.startswith(f"- {key}:")]
    assert len(lines) == 1
    assert description.lower() in lines[0].lower()


# ---------------------------------------------------------------------------
# TUI fork key: which steps it refuses
# ---------------------------------------------------------------------------

def test_fixture_covers_every_expected_step_type(base_dir):
    """Guards the tests below: a fixture drift could otherwise make them pass without testing the right steps."""
    app = RewindTUIApp(str(base_dir), CONV)
    expected = dict(TOOL_CALL_STEPS + NOTIFICATION_STEPS + PROMPT_STEPS + REPLY_STEPS)
    expected.update({step: category for step, category, _ in REFUSED_STEPS})
    expected[15] = "subagent_report"
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
    clone = only_child_of(base_dir, fork_id)
    expected_ids = lambda payload: payload.replace(CONV.encode(), fork_id.encode()).replace(CHILD.encode(), clone.encode())  # noqa: E731
    assert [i for i in range(step) if step_payload(base_dir, fork_id, i) != expected_ids(step_payload(base_dir, CONV, i))] == []
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
