"""
Tests for bugs found by reviewing the rewind engine. Each test names the wrong behaviour it guards against.

The session fixture and the Protobuf encoder are shared with test_forks.py, and every check inspects the real
files and databases a command leaves behind.
"""

import json
from pathlib import Path

import pytest

from rewinding.Antigravity_Rewind import detect_step_info
from test_forks import (
    CHILD,
    CONV,
    base_dir,  # noqa: F401 (a fixture)
    block_network,  # noqa: F401 (a fixture, active for every test here)
    bytes_field,
    int_field,
    open_db,
    run_cli,
    snapshot,
    system_message_payload,
)

SYSTEM_STEP = 18
TASK_SENDER = f"{CONV}/task-2"
MESSAGE_ID = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"


def set_step_payload(base: Path, conv_id: str, idx: int, payload: bytes) -> None:
    with open_db(base / "conversations" / f"{conv_id}.db") as conn:
        conn.execute("UPDATE steps SET step_payload = ? WHERE idx = ?", (payload, idx))


def write_message_file(base: Path, msg_id: str, sender: str, conv_id: str = CONV) -> Path:
    messages = base / "brain" / conv_id / ".system_generated" / "messages"
    messages.mkdir(exist_ok=True)
    path = messages / f"{msg_id}.json"
    path.write_text(json.dumps({"id": msg_id, "sender": sender, "recipient": conv_id, "content": "text"}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# A system message is never taken for a message from another step
# ---------------------------------------------------------------------------

UNIDENTIFIED_SYSTEM_MESSAGES = [
    pytest.param(int_field(1, 101) + int_field(4, 3) + bytes_field(5, bytes_field(12, b"meta")), id="real-no-message-field"),
    pytest.param(system_message_payload(MESSAGE_ID, "system", "platform notice"), id="real-system-sender"),
    pytest.param(b"", id="empty"),
    pytest.param(b"\x00", id="null-byte"),
    pytest.param(b"\xa2\x07", id="truncated"),
    pytest.param(b"   ", id="whitespace"),
]


@pytest.mark.parametrize("payload", UNIDENTIFIED_SYSTEM_MESSAGES)
@pytest.mark.parametrize("other_sender", [CHILD, TASK_SENDER])
def test_system_message_ignores_message_files_of_other_steps(base_dir, monkeypatch, capsys, payload, other_sender):
    """Any task or subagent message in the session used to turn a plain system message into a report from it."""
    set_step_payload(base_dir, CONV, SYSTEM_STEP, payload)
    write_message_file(base_dir, "ffffffff-ffff-ffff-ffff-ffffffffffff", other_sender)
    before = snapshot(base_dir)

    assert detect_step_info(str(base_dir), CONV, SYSTEM_STEP) == ("system_message", "system")
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, str(SYSTEM_STEP))

    assert exit_info.value.code == 1
    assert capsys.readouterr().err.splitlines()[0] == f"Error: Step {SYSTEM_STEP} is a Platform System Message."
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("sender, expected", [
    (CHILD, ("subagent_report", CHILD)),
    (TASK_SENDER, ("task_notification", TASK_SENDER)),
    ("system", ("system_message", "system")),
    (CONV, ("system_message", "system")),
])
def test_message_without_a_sender_uses_its_own_message_file(base_dir, sender, expected):
    """A message that names its ID but no sender can still be identified from the file with that ID."""
    message = bytes_field(114, bytes_field(4, bytes_field(1, MESSAGE_ID.encode()) + bytes_field(4, b"text")))
    set_step_payload(base_dir, CONV, SYSTEM_STEP, message)
    write_message_file(base_dir, MESSAGE_ID, sender)
    write_message_file(base_dir, "ffffffff-ffff-ffff-ffff-ffffffffffff", CHILD if sender != CHILD else TASK_SENDER)

    assert detect_step_info(str(base_dir), CONV, SYSTEM_STEP) == expected


@pytest.mark.parametrize("content", ["", "not json", "[]", "null", "\x00"])
def test_damaged_own_message_file_leaves_the_step_unidentified(base_dir, content):
    message = bytes_field(114, bytes_field(4, bytes_field(1, MESSAGE_ID.encode())))
    set_step_payload(base_dir, CONV, SYSTEM_STEP, message)
    path = write_message_file(base_dir, MESSAGE_ID, CHILD)
    path.write_text(content, encoding="utf-8")

    assert detect_step_info(str(base_dir), CONV, SYSTEM_STEP) == ("system_message", "system")
