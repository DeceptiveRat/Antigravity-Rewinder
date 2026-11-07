"""
Tests for bugs found by reviewing the rewind engine. Each test names the wrong behaviour it guards against.

The session fixture and the Protobuf encoder are shared with test_forks.py, and every check inspects the real
files and databases a command leaves behind.
"""

import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from rewinding.Antigravity_Rewind import count_or_truncate_chunks, detect_step_info
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


# ---------------------------------------------------------------------------
# Chunk folders are one transcript cut into pieces of a fixed size
# ---------------------------------------------------------------------------

CHUNK_SIZE = 102400


def chunk_folder(base: Path, conv_id: str = CONV, name: str = "transcript_full") -> Path:
    return base / "brain" / conv_id / ".system_generated" / "logs" / "chunks" / name


def write_transcript_as_chunks(base: Path, records: list[dict], conv_id: str = CONV) -> bytes:
    """Writes a transcript the way Antigravity does: a piece can start in the middle of a line or a character."""
    data = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")
    logs = base / "brain" / conv_id / ".system_generated" / "logs"
    (logs / "transcript_full.jsonl").write_bytes(data)
    (logs / "transcript.jsonl").write_bytes(data)
    for name in ("transcript_full", "transcript"):
        folder = chunk_folder(base, conv_id, name)
        for old in folder.glob("*.jsonl"):
            old.unlink()
        for number, start in enumerate(range(0, len(data), CHUNK_SIZE)):
            (folder / f"{number:08d}.jsonl").write_bytes(data[start:start + CHUNK_SIZE])
    return data


def assert_chunks_match_transcript(base: Path, conv_id: str = CONV) -> None:
    """The layout found in every real session: the pieces, in order, are exactly the transcript."""
    logs = base / "brain" / conv_id / ".system_generated" / "logs"
    for name in ("transcript_full", "transcript"):
        data = (logs / f"{name}.jsonl").read_bytes()
        files = sorted(chunk_folder(base, conv_id, name).glob("*.jsonl"))
        assert b"".join(path.read_bytes() for path in files) == data
        assert [path.name for path in files] == [f"{i:08d}.jsonl" for i in range((len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE)]
        assert [path.stat().st_size for path in files[:-1]] == [CHUNK_SIZE] * (len(files) - 1)


def long_session_records(base: Path) -> list[dict]:
    """The fixture's transcript, padded with Korean text so that pieces start inside a character."""
    path = base / "brain" / CONV / ".system_generated" / "logs" / "transcript_full.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["note"] = "가" * 9000
    return records


def starts_inside_a_character(folder: Path) -> list[str]:
    return [path.name for path in sorted(folder.glob("*.jsonl")) if path.read_bytes()[:1] and path.read_bytes()[0] & 0xC0 == 0x80]


@pytest.mark.parametrize("args, last_kept", [
    (["19"], 18),
    (["16"], 15),
    (["1", "--replace-text", "NEW TEXT"], 2),
    (["3", "--replace-old", "command", "--replace-new", "NEW TEXT"], 4),
    (["11", "--task-status", "failed"], 12),
    (["17", "--task-status", "failed"], 17),
    (["15", "--subagent-exit", "keep-alive", "--replace-text", "NEW TEXT"], 15),
])
def test_commands_survive_chunks_split_inside_a_character(base_dir, monkeypatch, args, last_kept):
    """Reading each piece as text crashed partway on real sessions, after the database had already been changed."""
    records = long_session_records(base_dir)
    write_transcript_as_chunks(base_dir, records)
    assert starts_inside_a_character(chunk_folder(base_dir)) != []

    run_cli(monkeypatch, base_dir, *args)

    assert_chunks_match_transcript(base_dir)
    logs = base_dir / "brain" / CONV / ".system_generated" / "logs"
    lines = (logs / "transcript_full.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step_index"] for line in lines] == list(range(last_kept + 1))


def test_fork_survives_chunks_split_inside_a_character(base_dir, monkeypatch):
    write_transcript_as_chunks(base_dir, long_session_records(base_dir))
    before = snapshot(base_dir / "brain" / CONV)

    run_cli(monkeypatch, base_dir, "1", "--replace-text", "NEW TEXT", "--fork", "b")

    [fork_id] = [path.name for path in (base_dir / "brain").iterdir() if path.name not in (CONV, CHILD)]
    assert_chunks_match_transcript(base_dir, fork_id)
    assert snapshot(base_dir / "brain" / CONV) == before


def test_dry_run_with_split_chunks_reports_counts_and_writes_nothing(base_dir, monkeypatch, capsys):
    write_transcript_as_chunks(base_dir, long_session_records(base_dir))
    before = snapshot(base_dir)

    run_cli(monkeypatch, base_dir, "19", "--dry-run")

    assert snapshot(base_dir) == before
    expected_kept = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in long_session_records(base_dir)[:19]).encode()
    file_count = len(list(chunk_folder(base_dir).glob("*.jsonl")))
    kept_files = (len(expected_kept) + CHUNK_SIZE - 1) // CHUNK_SIZE
    assert f"[DRY-RUN] Chunks to truncate: 6 lines (delete {(file_count - kept_files) * 2} empty chunk files)" in capsys.readouterr().out


@given(
    sizes=st.lists(st.integers(min_value=0, max_value=30000), min_size=1, max_size=12),
    cut=st.integers(min_value=0, max_value=12),
    character=st.sampled_from(["a", "가", "😀", " ", "\x0b"]),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_truncating_chunks_keeps_exactly_the_earlier_lines(tmp_path_factory, sizes, cut, character):
    """Metamorphic check against a plain filter of the lines, for any record sizes and split points."""
    folder = tmp_path_factory.mktemp("chunks")
    lines = [json.dumps({"step_index": i, "text": character * size}, ensure_ascii=False) + "\n" for i, size in enumerate(sizes)]
    data = "".join(lines).encode("utf-8")
    for number, start in enumerate(range(0, len(data), CHUNK_SIZE)):
        (folder / f"{number:08d}.jsonl").write_bytes(data[start:start + CHUNK_SIZE])

    removed, deleted = count_or_truncate_chunks(str(folder), cut, dry_run=False)

    kept = "".join(lines[:cut]).encode("utf-8")
    files = sorted(folder.glob("*.jsonl"))
    assert removed == max(0, len(lines) - cut)
    assert b"".join(path.read_bytes() for path in files) == kept
    assert [path.stat().st_size for path in files[:-1]] == [CHUNK_SIZE] * (len(files) - 1)
    assert deleted == (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE - (len(kept) + CHUNK_SIZE - 1) // CHUNK_SIZE


@pytest.mark.parametrize("damage", [b"\xff\xfe broken", b"\x00\x00", b"", b"   \n", b"{\"step_index\": "])
def test_damaged_chunk_bytes_are_kept_rather_than_crashing(tmp_path, damage):
    """A partial line is what Antigravity leaves while writing, so it must survive unchanged."""
    folder = tmp_path / "chunks"
    folder.mkdir()
    good = json.dumps({"step_index": 0}) + "\n"
    (folder / "00000000.jsonl").write_bytes(good.encode() + json.dumps({"step_index": 5}).encode() + b"\n" + damage)

    assert count_or_truncate_chunks(str(folder), 1, dry_run=False) == (1, 0)

    expected = good.encode() + (b"" if not damage.strip() else damage + b"\n")
    assert (folder / "00000000.jsonl").read_bytes() == expected
