"""
Tests for bugs found by reviewing the rewind engine. Each test names the wrong behaviour it guards against.

The session fixture and the Protobuf encoder are shared with test_forks.py, and every check inspects the real
files and databases a command leaves behind.
"""

import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from rewinding.Antigravity_Rewind import (
    count_or_truncate_chunks,
    detect_step_info,
    extract_subagent_msg_info,
    replace_in_result,
)
from test_forks import (
    CHILD,
    CONV,
    add_task_logs,
    base_dir,  # noqa: F401 (a fixture)
    block_network,  # noqa: F401 (a fixture, active for every test here)
    bytes_field,
    error_box,
    int_field,
    open_db,
    run_cli,
    snapshot,
    system_message_payload,
    task_logs,
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
    (["15", "--subagent-exit", "crash"], 16),
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


# ---------------------------------------------------------------------------
# Replacements change the result's text in every real format, and nothing else
# ---------------------------------------------------------------------------

METADATA = bytes_field(5, (
    bytes_field(4, bytes_field(3, b"a note about 1 and the original text"))
    + bytes_field(12, b"aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa")
    + bytes_field(20, bytes_field(1, b"bbbbbbbb-1111-1111-1111-bbbbbbbbbbbb") + bytes_field(4, CONV.encode()))
))
ARGUMENTS = bytes_field(1, bytes_field(1, b"CommandLine") + bytes_field(2, b"cat original 1"))
LABELS = bytes_field(5, b"run 1") + bytes_field(6, b"original") + bytes_field(9, bytes_field(1, b"original 1"))
WRAPPED_TYPE = b"type.googleapis.com/gemini_coder.Step"
OUTPUT_PATH = bytes_field(7, f"/brain/{CONV}/steps/1/output.txt".encode())
ID_IN_TEXT_FIELD = bytes_field(2, bytes_field(1, b"key") + bytes_field(2, b"cccccccc-1111-1111-1111-cccccccccccc"))
NUMBER_NOTE = (
    "The following code has been modified to include a line number before every line, in the format: "
    "<line_number>: <original_line>. Please note that any changes targeting the original code should remove "
    "the line number, colon, and leading space."
)
VIEW_HEAD = f"File Path: `/tmp/original.txt`\nTotal Lines: 12\nShowing lines 10 to 11\n{NUMBER_NOTE}\n"
VIEW_TAIL = "The above content does NOT show the entire file contents."


def result_step(observation: bytes, outside: bytes = b"") -> bytes:
    """A real result: the type and status, metadata, then the call's arguments and the observation in field 140."""
    return int_field(1, 132) + int_field(4, 3) + METADATA + outside + bytes_field(140, ARGUMENTS + bytes_field(2, observation))


def wrapped(inner: bytes) -> bytes:
    """A wrapped copy of the step, with its own metadata."""
    return bytes_field(6, bytes_field(1, WRAPPED_TYPE) + bytes_field(2, METADATA + inner))


def command_record(code: int, output: str) -> bytes:
    """The record in the newest sessions: 1 command, 2 working folder, 3 exit code, 4 output."""
    record = bytes_field(1, b"cat original 1") + bytes_field(2, b"/tmp/original") + int_field(3, code) + bytes_field(4, output.encode())
    return bytes_field(3, LABELS + bytes_field(10, bytes_field(3, record)))


def older_command_record(output: str) -> bytes:
    return wrapped(bytes_field(28, bytes_field(21, bytes_field(1, output.encode()))))


def shape_plain(text: str) -> bytes:
    return result_step(bytes_field(1, text.encode()) + OUTPUT_PATH)


def shape_newest_command(text: str) -> bytes:
    return result_step(bytes_field(1, text.encode()) + command_record(0, text) + OUTPUT_PATH)


def shape_record_only(text: str) -> bytes:
    return result_step(command_record(0, text))


def shape_older_command(text: str) -> bytes:
    return result_step(bytes_field(1, text.encode()) + older_command_record(text) + OUTPUT_PATH)


def shape_search(text: str) -> bytes:
    return result_step(bytes_field(1, text.encode()) + wrapped(bytes_field(42, bytes_field(5, text.encode()))))


RESULT_SHAPES = [
    pytest.param(shape_plain, id="text-only"),
    pytest.param(shape_newest_command, id="newest-command"),
    pytest.param(shape_record_only, id="record-only"),
    pytest.param(shape_older_command, id="older-command"),
    pytest.param(shape_search, id="search"),
]
TOOL_STEPS = [pytest.param(1, 2, id="read"), pytest.param(3, 4, id="command"), pytest.param(5, 6, id="write")]
TIMESTAMPS = "Created At: 2026-10-19T05:10:00Z\nCompleted At: 2026-10-19T05:10:01Z\n\n"


def transcript_files(base: Path, conv_id: str = CONV) -> list[Path]:
    logs = base / "brain" / conv_id / ".system_generated" / "logs"
    return [logs / "transcript_full.jsonl", logs / "transcript.jsonl", *sorted((logs / "chunks").glob("*/*.jsonl"))]


def store_result(base: Path, step: int, payload: bytes, shown_text: str, output_file: str | None = None) -> None:
    """Stores a result, with the text transcripts show for it, and optionally its output file."""
    set_step_payload(base, CONV, step, payload)
    for path in transcript_files(base):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in records:
            if record["step_index"] == step:
                record["content"] = shown_text
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    if output_file is not None:
        folder = base / "brain" / CONV / ".system_generated" / "steps" / str(step)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "output.txt").write_bytes(output_file.encode())


def shown_texts(base: Path, step: int, conv_id: str = CONV) -> list[str]:
    """The result's text in every transcript and chunk file, which all have to agree."""
    return [
        record["content"]
        for path in transcript_files(base, conv_id)
        for record in map(json.loads, path.read_text(encoding="utf-8").splitlines())
        if record["step_index"] == step
    ]


def stored_payload(base: Path, step: int, conv_id: str = CONV) -> bytes:
    with open_db(base / "conversations" / f"{conv_id}.db") as conn:
        return conn.execute("SELECT step_payload FROM steps WHERE idx = ?", (step,)).fetchone()[0]


@pytest.mark.parametrize("call, result", TOOL_STEPS)
@pytest.mark.parametrize("shape", RESULT_SHAPES)
def test_whole_replacement_changes_every_copy_of_the_text(base_dir, monkeypatch, call, result, shape):
    """A result with no 140.2.1 failed outright, and others kept the original in their command record."""
    store_result(base_dir, result, shape("the original text"), TIMESTAMPS + "the original text", "wrapped: the original text")

    run_cli(monkeypatch, base_dir, str(call), "--replace-text", "NEW TEXT")

    assert stored_payload(base_dir, result) == shape("NEW TEXT")
    assert shown_texts(base_dir, result) == [TIMESTAMPS + "NEW TEXT"] * 4


@pytest.mark.parametrize("shape", RESULT_SHAPES)
@pytest.mark.parametrize("search, replacement", [("1", "9"), ("original", "CHANGED"), ("text", "t"), ("a", "bbbbbbbbbbbb")])
def test_search_text_changes_only_the_result_text(base_dir, monkeypatch, shape, search, replacement):
    """A search for "1" used to rewrite the conversation ID, and the arguments and metadata with it."""
    text = "value 1 and the original text"
    store_result(base_dir, 2, shape(text), TIMESTAMPS + text)

    run_cli(monkeypatch, base_dir, "1", "--replace-old", search, "--replace-new", replacement)

    assert stored_payload(base_dir, 2).count(CONV.encode()) == shape(text).count(CONV.encode())
    assert stored_payload(base_dir, 2) == shape(text.replace(search, replacement))
    assert shown_texts(base_dir, 2) == [TIMESTAMPS.replace(search, replacement) + text.replace(search, replacement)] * 4


@pytest.mark.parametrize("search", [CONV, "aaaaaaaa-1111", "CommandLine", "a note about", "gemini_coder", "output.txt"])
def test_text_found_only_outside_the_result_is_reported_as_not_found(base_dir, monkeypatch, capsys, search):
    store_result(base_dir, 2, shape_plain("the visible text"), TIMESTAMPS + "the visible text")
    before = snapshot(base_dir)

    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, "1", "--replace-old", search, "--replace-new", "CHANGED")

    assert exit_info.value.code == 1
    assert f"Could not find '{search}' in step 2, so the replacement was not applied. Nothing was changed." in capsys.readouterr().err
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("text", [
    "\n\nabcdefghij",
    "\n\x0babc 12345678",
    "b\x05hello",
    "\x12\x03abc",
    "\n",
    "$" * 36,
])
def test_text_that_parses_as_a_message_is_still_treated_as_text(base_dir, monkeypatch, text):
    """Its length byte used to be recalculated as if it belonged to a nested message, changing a character."""
    store_result(base_dir, 2, shape_plain(text + " tail"), TIMESTAMPS + text + " tail")
    run_cli(monkeypatch, base_dir, "1", "--replace-old", "tail", "--replace-new", "longer tail")
    assert stored_payload(base_dir, 2) == shape_plain(text + " longer tail")


@pytest.mark.parametrize("payload", [
    pytest.param(result_step(LABELS and bytes_field(3, LABELS)), id="labels-but-no-text"),
    pytest.param(int_field(1, 132) + int_field(4, 3) + METADATA, id="metadata-only"),
    pytest.param(result_step(b""), id="empty-observation"),
])
def test_result_without_stored_text_is_refused_and_keeps_its_structure(base_dir, monkeypatch, capsys, payload):
    """The first nested message used to be taken for the text, and was replaced by the new text."""
    store_result(base_dir, 2, payload, TIMESTAMPS + "text only in the transcript")
    before = snapshot(base_dir)

    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, "1", "--replace-text", "NEW TEXT")

    assert exit_info.value.code == 1
    assert "Could not find 'text only in the transcript' in step 2" in capsys.readouterr().err
    assert snapshot(base_dir) == before


@given(
    before=st.text(min_size=0, max_size=30),
    search=st.text(min_size=1, max_size=8),
    after=st.text(min_size=0, max_size=30),
    replacement=st.text(min_size=1, max_size=8),
)
@settings(max_examples=100, deadline=None)
def test_replacing_in_a_result_equals_replacing_in_its_text(before, search, after, replacement):
    """Whatever the text, the step is the same as one built from the replaced text, so no other byte moved."""
    text = before + search + after
    new_payload, count = replace_in_result(shape_newest_command(text), search.encode(), replacement.encode())
    expected = text.replace(search, replacement)
    assert new_payload == shape_newest_command(expected)
    assert count == text.count(search) * 2


# ---------------------------------------------------------------------------
# File views number their lines, which a plain search cannot see through
# ---------------------------------------------------------------------------

def view_text(*lines: str, first: int = 10) -> str:
    return VIEW_HEAD + "".join(f"{first + i}: {line}\n" for i, line in enumerate(lines)) + VIEW_TAIL


def view_shape(text: str, raw: str) -> bytes:
    return result_step(bytes_field(1, text.encode()) + wrapped(bytes_field(14, bytes_field(4, raw.encode()))))


def test_whole_replacement_of_a_file_view_changes_what_the_agent_reads(base_dir, monkeypatch):
    """Only the raw copy used to change, so the numbered text the agent reads kept the original."""
    store_result(base_dir, 2, view_shape(view_text("original one", "original two"), "original one\noriginal two"),
                 TIMESTAMPS + view_text("original one", "original two"))

    run_cli(monkeypatch, base_dir, "1", "--replace-text", "NEW ONE\nNEW TWO\nNEW THREE")

    new_view = view_text("NEW ONE", "NEW TWO", "NEW THREE")
    assert stored_payload(base_dir, 2) == view_shape(new_view, "NEW ONE\nNEW TWO\nNEW THREE")
    assert shown_texts(base_dir, 2) == [TIMESTAMPS + new_view] * 4


@pytest.mark.parametrize("search, replacement, expected_lines", [
    ("original one", "CHANGED", ["CHANGED", "original two"]),
    ("one\noriginal", "A\nB\nC", ["original A", "B", "C two"]),
    ("original one\noriginal two", "ONE LINE", ["ONE LINE"]),
])
def test_search_in_a_file_view_keeps_the_numbering_correct(base_dir, monkeypatch, search, replacement, expected_lines):
    """Text across lines was not found in the numbered copy, and the transcript was replaced by the search result."""
    raw = "original one\noriginal two"
    store_result(base_dir, 2, view_shape(view_text("original one", "original two"), raw), TIMESTAMPS + view_text("original one", "original two"))

    run_cli(monkeypatch, base_dir, "1", "--replace-old", search, "--replace-new", replacement)

    new_view = view_text(*expected_lines)
    assert stored_payload(base_dir, 2) == view_shape(new_view, raw.replace(search, replacement))
    assert shown_texts(base_dir, 2) == [TIMESTAMPS + new_view] * 4


# ---------------------------------------------------------------------------
# Simulated errors follow the format of the session's real results
# ---------------------------------------------------------------------------

NEW_FORMAT = "Command exited with code 0.\n\nCommand output:\nall good\n"
OLD_FORMAT = "\nThe command exited with code 0.\nOutput:\nall good\r\n\n"


@pytest.mark.parametrize("options, code, output", [
    (["--simulate-error", "3", "--error-output", "disk full"], 3, "disk full"),
    (["--simulate-error", "7"], 7, ""),
    (["--simulate-error"], 1, ""),
    (["--simulate-error", "2:not found"], 2, "not found"),
])
def test_simulated_error_on_the_newest_format(base_dir, monkeypatch, options, code, output):
    """The record kept exit code 0 and the real output, and the wording differed from every other result."""
    store_result(base_dir, 4, result_step(bytes_field(1, NEW_FORMAT.encode()) + command_record(0, "all good") + OUTPUT_PATH), TIMESTAMPS + NEW_FORMAT, NEW_FORMAT)

    run_cli(monkeypatch, base_dir, "3", *options)

    first_line = f"Command exited with code {code}."
    text = f"{first_line}\n\nCommand output:\n{output}\n" if output else f"{first_line}\n"
    assert stored_payload(base_dir, 4) == result_step(bytes_field(1, text.encode()) + command_record(code, output) + OUTPUT_PATH)
    assert shown_texts(base_dir, 4) == [TIMESTAMPS + text.strip()] * 4


def test_simulated_error_when_only_the_record_holds_the_output(base_dir, monkeypatch):
    """This format had no text field to rewrite, so the command stopped with an error."""
    store_result(base_dir, 4, result_step(command_record(0, "all good")), TIMESTAMPS + NEW_FORMAT, NEW_FORMAT)

    run_cli(monkeypatch, base_dir, "3", "--simulate-error", "3", "--error-output", "disk full")

    assert stored_payload(base_dir, 4) == result_step(command_record(3, "disk full"))


def test_simulated_error_keeps_the_older_format(base_dir, monkeypatch):
    original = result_step(bytes_field(1, OLD_FORMAT.encode()) + older_command_record("all good\r\n") + OUTPUT_PATH)
    store_result(base_dir, 4, original, TIMESTAMPS + OLD_FORMAT, OLD_FORMAT)

    run_cli(monkeypatch, base_dir, "3", "--simulate-error", "3", "--error-output", "disk full")

    new_text = "\nThe command exited with code 3.\nOutput:\ndisk full\r\n\n"
    record = wrapped(bytes_field(28, bytes_field(21, bytes_field(1, b"disk full\r\n")) + int_field(6, 3) + int_field(11, 1)))
    assert stored_payload(base_dir, 4) == result_step(bytes_field(1, new_text.encode()) + record + OUTPUT_PATH)


# ---------------------------------------------------------------------------
# Changing a subagent's report keeps its stored history readable
# ---------------------------------------------------------------------------

REPORT_STEP = 15


def child_steps(base: Path) -> dict[int, bytes]:
    with open_db(base / "conversations" / f"{CHILD}.db") as conn:
        return dict(conn.execute("SELECT idx, step_payload FROM steps ORDER BY idx"))


@pytest.mark.parametrize("old, new", [("Worker", "WORKER_MUCH_LONGER"), ("Worker", "W"), ("analysis", "한글 분석")])
def test_changing_a_report_keeps_child_steps_readable(base_dir, monkeypatch, old, new):
    """Bytes were swapped without fixing the length before them, which made the steps unreadable."""
    text = f"{old} finished early"
    set_step_payload(base_dir, CHILD, 1, shape_plain(text))
    report = f"{old} finished the analysis"
    set_step_payload(base_dir, CONV, REPORT_STEP, system_message_payload(MESSAGE_ID, CHILD, report, child_step=2))

    run_cli(monkeypatch, base_dir, str(REPORT_STEP), "--subagent-exit", "keep-alive", "--replace-old", old, "--replace-new", new)

    assert child_steps(base_dir) == {0: b"child step", 1: shape_plain(text.replace(old, new)), 2: b"child step"}


def test_changing_a_report_leaves_child_metadata_alone(base_dir, monkeypatch):
    """A search for "1" must not rewrite the IDs in the child's stored steps."""
    set_step_payload(base_dir, CHILD, 1, shape_plain("value 1"))
    set_step_payload(base_dir, CONV, REPORT_STEP, system_message_payload(MESSAGE_ID, CHILD, "report 1", child_step=2))

    run_cli(monkeypatch, base_dir, str(REPORT_STEP), "--subagent-exit", "keep-alive", "--replace-old", "1", "--replace-new", "9")

    assert child_steps(base_dir)[1] == shape_plain("value 9")


@pytest.mark.parametrize("damage", [b"not a database at all", b"\x00" * 200])
def test_damaged_child_database_stops_before_anything_changes(base_dir, monkeypatch, capsys, damage):
    """The failure used to be hidden, so the parent reported success with the child left as it was."""
    (base_dir / "conversations" / f"{CHILD}.db").write_bytes(damage)
    before = snapshot(base_dir)

    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, str(REPORT_STEP), "--subagent-exit", "crash")

    assert exit_info.value.code == 1
    assert "Error: file is not a database" in capsys.readouterr().err
    assert snapshot(base_dir) == before


@pytest.mark.parametrize("old, new", [
    ("Worker", "WorkerWorker"),
    ("a", "aa"),
    ("the", "the the"),
])
def test_report_replacement_is_applied_once(base_dir, monkeypatch, old, new):
    """When the new text held the old text, a second pass replaced it again, so the agent read something else."""
    report = "Worker finished the analysis"
    set_step_payload(base_dir, CONV, REPORT_STEP, system_message_payload(MESSAGE_ID, CHILD, report, child_step=2))

    run_cli(monkeypatch, base_dir, str(REPORT_STEP), "--subagent-exit", "keep-alive", "--replace-old", old, "--replace-new", new)

    expected = report.replace(old, new)
    assert stored_payload(base_dir, REPORT_STEP) == system_message_payload(MESSAGE_ID, CHILD, expected, child_step=2)


@pytest.mark.parametrize("search", [CHILD[:8], MESSAGE_ID[:8], "MESSAGE_PRIORITY", "timestamp"])
def test_report_search_text_outside_the_report_is_not_found(base_dir, monkeypatch, capsys, search):
    """Text that only appeared in the message's IDs or header passed the check, and nothing was changed."""
    set_step_payload(base_dir, CONV, REPORT_STEP, system_message_payload(MESSAGE_ID, CHILD, "Worker report", child_step=2))
    before = snapshot(base_dir)

    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, str(REPORT_STEP), "--subagent-exit", "keep-alive", "--replace-old", search, "--replace-new", "CHANGED")

    assert exit_info.value.code == 1
    assert f"Target string '{search}' not found in Step {REPORT_STEP} subagent report." in capsys.readouterr().err
    assert snapshot(base_dir) == before


# ---------------------------------------------------------------------------
# Half of a replacement is refused before anything changes
# ---------------------------------------------------------------------------

HALF_REPLACEMENTS = [
    pytest.param(["--replace-old", "Worker"], "empty", id="old-only"),
    pytest.param(["--replace-old", "Worker", "--replace-new", ""], "empty", id="old-with-empty-new"),
    pytest.param(["--replace-new", "X"], "no-search", id="new-only"),
]
STEPS_WITH_HALF_REPLACEMENT_BUG = [
    pytest.param("17", [], 17, id="task-notification"),
    pytest.param("11", [], 12, id="manage-task"),
    pytest.param("15", ["--subagent-exit", "keep-alive"], 15, id="report-keep-alive"),
    pytest.param("15", ["--subagent-exit", "crash"], 15, id="report-crash"),
    pytest.param("15", ["--subagent-exit", "complete"], 15, id="report-complete"),
]


def half_replacement_error(kind: str, step: int) -> str:
    if kind == "empty":
        return f"The replacement text is empty, so step {step} would not change. Nothing was changed."
    return "--replace-new needs --replace-old, the text to search for. Nothing was changed."


@pytest.mark.parametrize("step, extra, shown_step", STEPS_WITH_HALF_REPLACEMENT_BUG)
@pytest.mark.parametrize("options, kind", HALF_REPLACEMENTS)
@pytest.mark.parametrize("fork", [[], ["--fork", "b"]], ids=["in-place", "fork"])
def test_half_a_replacement_is_refused(base_dir, monkeypatch, capsys, step, extra, shown_step, options, kind, fork):
    """These deleted later steps and reported success, and --replace-old alone turned a failed task into a success."""
    before = snapshot(base_dir)

    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, base_dir, step, *extra, *options, *fork)

    assert exit_info.value.code == 1
    assert capsys.readouterr().err == error_box(half_replacement_error(kind, shown_step))
    assert snapshot(base_dir) == before


def test_failed_task_is_not_rewritten_as_a_success(base_dir, monkeypatch, capsys):
    """The exact case found by the review: a search text alone produced "exited with code 0"."""
    failed = f'Task id "{TASK_SENDER}" finished with result:\n\nThe command exited with code 7.'
    original = system_message_payload(MESSAGE_ID, TASK_SENDER, failed)
    set_step_payload(base_dir, CONV, 17, original)

    with pytest.raises(SystemExit):
        run_cli(monkeypatch, base_dir, "17", "--replace-old", "foo")

    assert stored_payload(base_dir, 17) == original


# ---------------------------------------------------------------------------
# Task-finished messages say what was asked, and keep the rest of the step
# ---------------------------------------------------------------------------

def notification_text(base: Path, step: int) -> str:
    return extract_subagent_msg_info(stored_payload(base, step))[2]


@pytest.mark.parametrize("options, expected", [
    (["--task-status", "running"], f'Task id "{CONV}/task-4" is currently running.'),
    (["--task-status", "RUNNING"], f'Task id "{CONV}/task-4" is currently running.'),
    (["--task-status", "running", "--task-exit-code", "3"], f'Task id "{CONV}/task-4" is currently running.'),
    (["--task-status", "finished"], f'Task id "{CONV}/task-4" finished with result:\n\nThe command exited with code 0.'),
    (["--task-status", "failed"], f'Task id "{CONV}/task-4" finished with result:\n\nThe command exited with code 1.'),
    (["--task-status", "failed", "--task-exit-code", "5"], f'Task id "{CONV}/task-4" finished with result:\n\nThe command exited with code 5.'),
    (["--task-exit-code", "0"], f'Task id "{CONV}/task-4" finished with result:\n\nThe command exited with code 0.'),
    (["--task-status", "cancelled"], f'Task id "{CONV}/task-4" has been cancelled by the user.'),
])
def test_task_status_is_worded_as_asked(base_dir, monkeypatch, options, expected):
    """"running" used to be written as a task that had exited with code 1."""
    run_cli(monkeypatch, base_dir, "17", *options)
    assert notification_text(base_dir, 17) == expected
UNKNOWN_FIELD = bytes_field(77, b"a field this tool does not know")
HEADER = "[Message] timestamp=2026-10-19T05:10:00Z sender={} priority=MESSAGE_PRIORITY_HIGH content={}"


def real_notification(content: str) -> bytes:
    """A real message: step metadata, three copies of the text, and fields this tool has no use for."""
    message = (bytes_field(1, MESSAGE_ID.encode()) + bytes_field(2, CONV.encode()) + bytes_field(3, TASK_SENDER.encode())
               + bytes_field(4, content.encode()) + int_field(5, 1) + UNKNOWN_FIELD)
    render = bytes_field(1, b"System Message") + bytes_field(10, bytes_field(1, content.encode())) + UNKNOWN_FIELD
    details = bytes_field(1, HEADER.format(TASK_SENDER, content).encode()) + bytes_field(2, render) + bytes_field(4, message)
    return int_field(1, 101) + int_field(4, 3) + METADATA + bytes_field(114, details) + UNKNOWN_FIELD


@pytest.mark.parametrize("options, expected", [
    (["--replace-text", "NEW MESSAGE"], "NEW MESSAGE"),
    (["--replace-old", "original", "--replace-new", "CHANGED"], "the CHANGED message"),
    (["--task-status", "running"], f'Task id "{TASK_SENDER}" is currently running.'),
])
def test_editing_a_notification_changes_only_its_text(base_dir, monkeypatch, options, expected):
    """The step was rebuilt from scratch, which replaced its timestamps and IDs and dropped fields."""
    set_step_payload(base_dir, CONV, 17, real_notification("the original message"))

    run_cli(monkeypatch, base_dir, "17", *options)

    assert stored_payload(base_dir, 17) == real_notification(expected)


def test_notification_log_option_alone_leaves_the_step_exactly_as_it_was(base_dir, monkeypatch):
    """The step was rebuilt here too, although the option is only about the log."""
    add_task_logs(base_dir)
    original = real_notification("the original message")
    set_step_payload(base_dir, CONV, 17, original)

    run_cli(monkeypatch, base_dir, "17", "--task-output", "task-4=partial", "--keep-task-log")

    assert stored_payload(base_dir, 17) == original


def test_notification_without_stored_text_is_still_written(base_dir, monkeypatch):
    """A real message can have no text in the database, so there is nothing to edit in place."""
    bare = int_field(1, 101) + int_field(4, 3) + METADATA
    shown = f"<SYSTEM_MESSAGE>\n{HEADER.format(f'{CONV}/task-4', 'old message')}\n</SYSTEM_MESSAGE>"
    store_result(base_dir, 17, bare, shown)

    run_cli(monkeypatch, base_dir, "17", "--replace-text", "NEW MESSAGE")

    message_id, sender, text = extract_subagent_msg_info(stored_payload(base_dir, 17))
    assert (sender, text) == (f"{CONV}/task-4", "NEW MESSAGE")
    assert message_id is not None


# ---------------------------------------------------------------------------
# Output files only change when the result's text is really in them
# ---------------------------------------------------------------------------

def output_file(base: Path, step: int) -> Path:
    return base / "brain" / CONV / ".system_generated" / "steps" / str(step) / "output.txt"


def test_partial_replacement_does_not_create_an_output_file(base_dir, monkeypatch):
    """A new output.txt held only the snippet, which a later whole replacement then took for the whole result."""
    store_result(base_dir, 2, shape_plain("ORIGINAL FILE TEXT"), TIMESTAMPS + "ORIGINAL FILE TEXT")

    run_cli(monkeypatch, base_dir, "1", "--replace-old", "ORIGINAL", "--replace-new", "CHANGED")

    assert not output_file(base_dir, 2).exists()
    assert stored_payload(base_dir, 2) == shape_plain("CHANGED FILE TEXT")


def test_whole_replacement_after_a_partial_one_leaves_nothing_behind(base_dir, monkeypatch):
    """The exact sequence from the review, which ended in WHOLE NEW RESULT FILE TEXT."""
    store_result(base_dir, 2, shape_plain("ORIGINAL FILE TEXT"), TIMESTAMPS + "ORIGINAL FILE TEXT")

    run_cli(monkeypatch, base_dir, "1", "--replace-old", "ORIGINAL", "--replace-new", "CHANGED")
    run_cli(monkeypatch, base_dir, "1", "--replace-text", "WHOLE NEW RESULT")

    assert stored_payload(base_dir, 2) == shape_plain("WHOLE NEW RESULT")
    assert shown_texts(base_dir, 2) == [TIMESTAMPS + "WHOLE NEW RESULT"] * 4


@pytest.mark.parametrize("output", ["unrelated output", "", "ORIGINAL only partly", "  \n"])
def test_partial_replacement_leaves_a_differing_output_file_alone(base_dir, monkeypatch, output):
    """A file that did not hold the search text was overwritten with only the new snippet."""
    store_result(base_dir, 2, shape_plain("SEARCHED FILE TEXT"), TIMESTAMPS + "SEARCHED FILE TEXT", output)

    run_cli(monkeypatch, base_dir, "1", "--replace-old", "SEARCHED", "--replace-new", "CHANGED")

    assert output_file(base_dir, 2).read_bytes() == output.encode()


def test_partial_replacement_updates_an_output_file_holding_the_text(base_dir, monkeypatch):
    store_result(base_dir, 2, shape_plain("line one\r\nline two\r\n"), TIMESTAMPS + "line one\r\nline two\r\n", "line one\r\nline two\r\n")

    run_cli(monkeypatch, base_dir, "1", "--replace-old", "two", "--replace-new", "2")

    assert output_file(base_dir, 2).read_bytes() == b"line one\r\nline 2\r\n"


@pytest.mark.parametrize("existing", [None, "stale text that differs", ""])
def test_whole_replacement_writes_the_output_file(base_dir, monkeypatch, existing):
    store_result(base_dir, 2, shape_plain("ORIGINAL"), TIMESTAMPS + "ORIGINAL", existing)
    output_file(base_dir, 2).parent.parent.mkdir(parents=True, exist_ok=True)

    run_cli(monkeypatch, base_dir, "1", "--replace-text", "WHOLE NEW RESULT")

    assert output_file(base_dir, 2).read_bytes() == b"WHOLE NEW RESULT"


def test_dry_run_counts_only_the_files_it_would_change(base_dir, monkeypatch, capsys):
    store_result(base_dir, 2, shape_plain("SEARCHED"), TIMESTAMPS + "SEARCHED", "unrelated output")
    before = snapshot(base_dir)

    run_cli(monkeypatch, base_dir, "1", "--replace-old", "SEARCHED", "--replace-new", "CHANGED", "--dry-run")

    assert "Step artifact files to update" not in capsys.readouterr().out
    assert snapshot(base_dir) == before


# ---------------------------------------------------------------------------
# A background launch is recognised by how its result starts, not by mentions of one
# ---------------------------------------------------------------------------

LOGS_ALL = {"task-2": "TASK 2 FINAL OUTPUT\n", "task-4": "TASK 4 FINAL OUTPUT\n"}
LOGS_WITHOUT_TASK_2 = {"task-4": "TASK 4 FINAL OUTPUT\n"}
LAUNCH_LINE = f"Tool is running as a background task with task id: {CONV}/task-2\nTask Description: pytest"
CREATED_ONLY = "Created At: 2026-10-19T05:10:00Z\n"
MARKER_148 = bytes_field(148, bytes_field(1, f"{CONV}/task-2".encode()) + bytes_field(4, b"pytest"))


@pytest.mark.parametrize("shown", [
    pytest.param(TIMESTAMPS + "Notes: read about a background task in task-2", id="mention"),
    pytest.param(TIMESTAMPS + "Command exited with code 0.\n\nCommand output:\nThe word task-2 and background task\n", id="in-output"),
    pytest.param(TIMESTAMPS + f"Command exited with code 0.\n\nCommand output:\n{LAUNCH_LINE}\n", id="launch-text-in-output"),
    pytest.param(TIMESTAMPS + f"grep found: {LAUNCH_LINE}", id="launch-text-mid-line"),
])
def test_output_mentioning_a_background_task_keeps_its_log(base_dir, monkeypatch, shown):
    """A phrase anywhere in the result counted as a launch, which deleted the log, even with --keep-task-log."""
    add_task_logs(base_dir)
    store_result(base_dir, 4, shape_plain(shown), shown)

    run_cli(monkeypatch, base_dir, "3", "--simulate-error", "1", "--keep-task-log")

    assert task_logs(base_dir, CONV) == LOGS_ALL


@pytest.mark.parametrize("payload, shown", [
    pytest.param(shape_plain("started"), CREATED_ONLY + LAUNCH_LINE, id="real-without-marker"),
    pytest.param(result_step(bytes_field(1, b"started"), MARKER_148), CREATED_ONLY + LAUNCH_LINE, id="real-with-marker"),
    pytest.param(result_step(bytes_field(1, b"started"), MARKER_148), "started task-2", id="marker-and-text-mismatch"),
])
def test_simulated_error_on_a_real_launch_removes_its_log(base_dir, monkeypatch, payload, shown):
    """97 of 128 real launches have no marker and no phrase in the database, so they went unnoticed."""
    add_task_logs(base_dir)
    store_result(base_dir, 4, payload, shown)

    run_cli(monkeypatch, base_dir, "3", "--simulate-error", "1", "--keep-task-log")

    assert task_logs(base_dir, CONV) == LOGS_WITHOUT_TASK_2


def test_simulated_error_on_a_launch_is_shown_as_a_finished_result(base_dir, monkeypatch):
    add_task_logs(base_dir)
    store_result(base_dir, 4, shape_plain("started"), CREATED_ONLY + LAUNCH_LINE)

    run_cli(monkeypatch, base_dir, "3", "--simulate-error", "1", "--keep-task-log")

    created = "2026-10-19T05:10:00Z"
    finished = f"Created At: {created}\nCompleted At: {created}\n\nThe command exited with code 1.\n"
    assert shown_texts(base_dir, 4) == [finished] * 4
