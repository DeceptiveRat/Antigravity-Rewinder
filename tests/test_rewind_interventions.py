import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
import pytest

import rewinding.Antigravity_Rewind as rewind_mod

decode_varint = rewind_mod.decode_varint
encode_varint = rewind_mod.encode_varint
parse_fields = rewind_mod.parse_fields
serialize_fields = rewind_mod.serialize_fields
replace_in_bytes = rewind_mod.replace_in_bytes
detect_step_info = rewind_mod.detect_step_info
extract_url_from_step = rewind_mod.extract_url_from_step
sync_or_prune_step_dirs = rewind_mod.sync_or_prune_step_dirs
execute_tool_intervene = rewind_mod.execute_tool_intervene
execute_kickstart_cleanup = rewind_mod.execute_kickstart_cleanup
main = rewind_mod.main


@pytest.fixture
def mock_intervention_env(tmp_path):
    """
    Constructs an isolated, multi-turn Antigravity session covering:
      - Step 0: Prompt (USER_INPUT = 14)
      - Step 1: Tool Call view_file (Category A, MODEL_RESPONSE = 15)
      - Step 2: Tool Result view_file (step_type = 132)
      - Step 3: Tool Call run_command (Category C, MODEL_RESPONSE = 15)
      - Step 4: Tool Result run_command (step_type = 132, with output.txt)
      - Step 5: Tool Call write_to_file (Category B, MODEL_RESPONSE = 15)
      - Step 6: Tool Result write_to_file (step_type = 5 / CODE_ACTION)
      - Step 7: Tool Call search_web (Category D, MODEL_RESPONSE = 15)
      - Step 8: Tool Result search_web (step_type = 132, with output.txt)
      - Step 9: Tool Call read_url_content (Category D, MODEL_RESPONSE = 15)
      - Step 10: Tool Result read_url_content (step_type = 132, output.txt & content.md)
      - Step 11: Tool Call view_file on content.md (MODEL_RESPONSE = 15)
      - Step 12: Tool Result view_file on content.md (step_type = 132)
      - Step 13: Kickstart prompt step containing sentinel token __AGY_KICKSTART__ (USER_INPUT = 14)
      - Step 14: Model response after kickstart (MODEL_RESPONSE = 15)
    """
    gemini_root = tmp_path / ".gemini"
    base_dir = gemini_root / "antigravity-cli"
    base_dir.mkdir(parents=True)

    conv_id = "test-intervene-conv-1234"

    # 1. Database setup
    convs_dir = base_dir / "conversations"
    convs_dir.mkdir()
    db_path = convs_dir / f"{conv_id}.db"

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, status INTEGER, step_payload BLOB)")
    cur.execute("CREATE TABLE gen_metadata (idx INTEGER PRIMARY KEY, data BLOB)")
    cur.execute("CREATE TABLE executor_metadata (idx INTEGER PRIMARY KEY, data BLOB)")
    cur.execute("CREATE TABLE parent_references (idx INTEGER PRIMARY KEY, data BLOB)")
    cur.execute("CREATE TABLE battle_mode_infos (idx INTEGER PRIMARY KEY, data BLOB)")

    # Construct protobuf-compatible bytes for payloads
    def make_proto_payload(text: str) -> bytes:
        b = text.encode("utf-8")
        # Field 14, wire type 2: outer message
        # Inside: Field 4, wire type 2: raw string
        sub = serialize_fields([(4, 2, b)])
        return serialize_fields([(14, 2, sub)])

    steps_data = [
        (0, 14, 1, b"User Prompt: run diagnostics and search docs"),
        (1, 15, 1, b"call view_file config.json"),
        (2, 132, 3, make_proto_payload("DEBUG=False\nPORT=8080")),
        (3, 15, 1, b"call run_command python3 --version"),
        (4, 132, 3, make_proto_payload("Python 3.12.3")),
        (5, 15, 1, b"call write_to_file greeting.txt"),
        (6, 5, 3, make_proto_payload("File created successfully: greeting.txt")),
        (7, 15, 1, b"call search_web Antigravity CLI"),
        (8, 132, 3, make_proto_payload("Antigravity CLI is an agentic development tool.")),
        (9, 15, 1, b"call read_url_content https://example.com/docs"),
        (10, 132, 3, make_proto_payload("Title: Example Docs\n\nOG Description: Official documentation\n\nThe full content of the article at https://example.com/docs has been saved to: /mock/path/steps/10/content.md\n\nYou can use the view_file tool to read specific sections if needed.")),
        (11, 15, 1, b"call view_file content.md"),
        (12, 132, 3, make_proto_payload("# Example Docs Header\nDocumentation body")),
        (13, 14, 1, b"__AGY_KICKSTART__ Continue"),
        (14, 15, 1, b"Acknowledged kickstart"),
    ]

    for idx, stype, status, payload in steps_data:
        cur.execute("INSERT INTO steps VALUES (?, ?, ?, ?)", (idx, stype, status, payload))
        cur.execute("INSERT INTO parent_references VALUES (?, ?)", (idx, b"parent-ref"))
        cur.execute("INSERT INTO battle_mode_infos VALUES (?, ?)", (idx, b"battle-info"))

    for m_idx in range(7):
        cur.execute("INSERT INTO gen_metadata VALUES (?, ?)", (m_idx, b"gen-meta"))
        cur.execute("INSERT INTO executor_metadata VALUES (?, ?)", (m_idx, b"exec-meta"))

    conn.commit()
    conn.close()

    # 2. Summaries DB and proto cache
    sum_db = base_dir / "conversation_summaries.db"
    s_conn = sqlite3.connect(str(sum_db))
    s_cur = s_conn.cursor()
    s_cur.execute("""
        CREATE TABLE conversation_summaries (
            conversation_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            preview TEXT NOT NULL DEFAULT '',
            step_count INTEGER NOT NULL DEFAULT 0,
            last_user_input_step_index INTEGER NOT NULL DEFAULT -1,
            status TEXT NOT NULL DEFAULT 'CASCADE_RUN_STATUS_IDLE',
            not_fully_idle INTEGER NOT NULL DEFAULT 0,
            raw_summary BLOB
        )
    """)
    s_cur.execute(
        "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (conv_id, "Intervention Test Session", "Preview text", 15, 13, "CASCADE_RUN_STATUS_IDLE", 0, b"raw-summary")
    )
    s_conn.commit()
    s_conn.close()

    proto_cache = base_dir / "jetbox_summaries_proto.pb"
    proto_cache.write_bytes(b"mock-protobuf-cache")

    presence_lock = base_dir / "presence" / f"{conv_id}.lock"
    presence_lock.parent.mkdir(parents=True, exist_ok=True)
    presence_lock.write_bytes(b"locked")

    # 3. Transcripts and chunks
    logs_dir = base_dir / "brain" / conv_id / ".system_generated" / "logs"
    chunks_tf_dir = logs_dir / "chunks" / "transcript_full"
    chunks_tc_dir = logs_dir / "chunks" / "transcript"
    chunks_tf_dir.mkdir(parents=True, exist_ok=True)
    chunks_tc_dir.mkdir(parents=True, exist_ok=True)

    transcript_entries = [
        {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": "run diagnostics and search docs", "created_at": "2026-10-18T00:00:00Z"},
        {"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "tool_calls": [{"name": "view_file", "args": {"AbsolutePath": "config.json"}}], "created_at": "2026-10-18T00:00:01Z"},
        {"step_index": 2, "source": "MODEL", "type": "GENERIC", "status": "DONE", "content": "Created At: 2026-10-18T09:00:01+09:00\nCompleted At: 2026-10-18T09:00:02+09:00\nDEBUG=False\nPORT=8080", "created_at": "2026-10-18T00:00:02Z"},
        {"step_index": 3, "source": "MODEL", "type": "PLANNER_RESPONSE", "tool_calls": [{"name": "run_command", "args": {"CommandLine": "python3 --version"}}], "created_at": "2026-10-18T00:00:03Z"},
        {"step_index": 4, "source": "MODEL", "type": "GENERIC", "status": "DONE", "content": "Created At: 2026-10-18T09:00:03+09:00\nCompleted At: 2026-10-18T09:00:04+09:00\nPython 3.12.3", "created_at": "2026-10-18T00:00:04Z"},
        {"step_index": 5, "source": "MODEL", "type": "PLANNER_RESPONSE", "tool_calls": [{"name": "write_to_file", "args": {"TargetFile": "greeting.txt", "CodeContent": "Hello World"}}], "created_at": "2026-10-18T00:00:05Z"},
        {"step_index": 6, "source": "MODEL", "type": "CODE_ACTION", "status": "DONE", "content": "Created At: 2026-10-18T09:00:05+09:00\nCompleted At: 2026-10-18T09:00:06+09:00\nFile created successfully: greeting.txt", "created_at": "2026-10-18T00:00:06Z"},
        {"step_index": 7, "source": "MODEL", "type": "PLANNER_RESPONSE", "tool_calls": [{"name": "search_web", "args": {"query": "Antigravity CLI"}}], "created_at": "2026-10-18T00:00:07Z"},
        {"step_index": 8, "source": "MODEL", "type": "GENERIC", "status": "DONE", "content": "Created At: 2026-10-18T09:00:07+09:00\nCompleted At: 2026-10-18T09:00:08+09:00\nAntigravity CLI is an agentic development tool.", "created_at": "2026-10-18T00:00:08Z"},
        {"step_index": 9, "source": "MODEL", "type": "PLANNER_RESPONSE", "tool_calls": [{"name": "read_url_content", "args": {"Url": "https://example.com/docs"}}], "created_at": "2026-10-18T00:00:09Z"},
        {"step_index": 10, "source": "MODEL", "type": "GENERIC", "status": "DONE", "content": "Created At: 2026-10-18T09:00:09+09:00\nCompleted At: 2026-10-18T09:00:10+09:00\nTitle: Example Docs\n\nOG Description: Official documentation\n\nThe full content of the article at https://example.com/docs has been saved to: /mock/path/steps/10/content.md\n\nYou can use the view_file tool to read specific sections if needed.", "created_at": "2026-10-18T00:00:10Z"},
        {"step_index": 11, "source": "MODEL", "type": "PLANNER_RESPONSE", "tool_calls": [{"name": "view_file", "args": {"AbsolutePath": "content.md"}}], "created_at": "2026-10-18T00:00:11Z"},
        {"step_index": 12, "source": "MODEL", "type": "GENERIC", "status": "DONE", "content": "Created At: 2026-10-18T09:00:11+09:00\nCompleted At: 2026-10-18T09:00:12+09:00\n# Example Docs Header\nDocumentation body", "created_at": "2026-10-18T00:00:12Z"},
        {"step_index": 13, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": "__AGY_KICKSTART__ Continue", "created_at": "2026-10-18T00:00:13Z"},
        {"step_index": 14, "source": "MODEL", "type": "PLANNER_RESPONSE", "content": "Acknowledged kickstart", "created_at": "2026-10-18T00:00:14Z"},
    ]

    tf_path = logs_dir / "transcript_full.jsonl"
    tc_path = logs_dir / "transcript.jsonl"
    with open(tf_path, "w", encoding="utf-8") as f_full, open(tc_path, "w", encoding="utf-8") as f_comp:
        for item in transcript_entries:
            line = json.dumps(item, ensure_ascii=False) + "\n"
            f_full.write(line)
            f_comp.write(line)

    with open(chunks_tf_dir / "00000000.jsonl", "w", encoding="utf-8") as cf:
        for item in transcript_entries[:8]:
            cf.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(chunks_tf_dir / "00000001.jsonl", "w", encoding="utf-8") as cf:
        for item in transcript_entries[8:]:
            cf.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 4. Step artifact directories
    steps_dir = base_dir / "brain" / conv_id / ".system_generated" / "steps"
    (steps_dir / "4").mkdir(parents=True)
    (steps_dir / "4" / "output.txt").write_text("Python 3.12.3", encoding="utf-8")

    (steps_dir / "8").mkdir(parents=True)
    (steps_dir / "8" / "output.txt").write_text("Antigravity CLI is an agentic development tool.", encoding="utf-8")

    (steps_dir / "10").mkdir(parents=True)
    (steps_dir / "10" / "output.txt").write_text(
        "Title: Example Docs\n\nOG Description: Official documentation\n\nThe full content of the article at https://example.com/docs has been saved to: /mock/path/steps/10/content.md\n\nYou can use the view_file tool to read specific sections if needed.",
        encoding="utf-8"
    )
    (steps_dir / "10" / "content.md").write_text("# Original Example Docs Content\nBody paragraphs here.", encoding="utf-8")

    (steps_dir / "12").mkdir(parents=True)
    (steps_dir / "12" / "output.txt").write_text("# Example Docs Header\nDocumentation body", encoding="utf-8")

    # 5. History.jsonl
    hist_file = base_dir / "history.jsonl"
    hist_entries = [
        {"conversationId": conv_id, "timestamp": 1792281600000, "display": "run diagnostics and search docs"},
        {"conversationId": conv_id, "timestamp": 1792281613000, "display": "__AGY_KICKSTART__ Continue"},
        {"conversationId": "other-conv-uuid", "timestamp": 1792281620000, "display": "other conversation prompt"},
    ]
    with open(hist_file, "w", encoding="utf-8") as hf:
        for h in hist_entries:
            hf.write(json.dumps(h) + "\n")

    return {
        "base_dir": base_dir,
        "conv_id": conv_id,
        "db_path": db_path,
        "steps_dir": steps_dir,
        "tf_path": tf_path,
        "tc_path": tc_path,
        "hist_file": hist_file,
        "proto_cache": proto_cache,
        "presence_lock": presence_lock,
    }


# =========================================================================
# Module 1: Protobuf Varint & Byte Replacement Engine
# =========================================================================

def test_varint_encode_decode():
    test_integers = [0, 1, 2, 127, 128, 255, 300, 16383, 16384, 1000000]
    for val in test_integers:
        encoded = encode_varint(val)
        decoded, offset = decode_varint(encoded, 0)
        assert decoded == val
        assert offset == len(encoded)


def test_replace_in_bytes_resizing():
    # Construct message with field 14.4 containing text
    original_text = "ORIGINAL_STRING"
    sub = serialize_fields([(4, 2, original_text.encode("utf-8"))])
    msg = serialize_fields([(14, 2, sub)])

    # 1. Replace with longer string
    longer_text = "A_MUCH_LONGER_REPLACEMENT_STRING_12345"
    modified, count = replace_in_bytes(msg, original_text.encode("utf-8"), longer_text.encode("utf-8"))
    assert count == 1
    assert longer_text.encode("utf-8") in modified
    assert original_text.encode("utf-8") not in modified

    # Parse modified bytes to verify field 14.4 length is intact
    parsed = parse_fields(modified)
    assert len(parsed) == 1
    fn, wt, val = parsed[0]
    assert fn == 14 and wt == 2
    sub_parsed = parse_fields(val)
    assert sub_parsed[0][0] == 4
    assert sub_parsed[0][2].decode("utf-8") == longer_text

    # 2. Replace with shorter string
    shorter_text = "SHORT"
    modified_short, count2 = replace_in_bytes(modified, longer_text.encode("utf-8"), shorter_text.encode("utf-8"))
    assert count2 == 1
    assert shorter_text.encode("utf-8") in modified_short


# =========================================================================
# Module 2: Step Detection & Tool Classification (detect_step_info)
# =========================================================================

def test_detect_step_info_categories(mock_intervention_env):
    base_dir = str(mock_intervention_env["base_dir"])
    conv_id = mock_intervention_env["conv_id"]

    # Step 0: Prompt
    cat, detail = detect_step_info(base_dir, conv_id, 0)
    assert cat == "prompt"
    assert detail is None

    # Step 1: Read Tool (view_file)
    cat, detail = detect_step_info(base_dir, conv_id, 1)
    assert cat == "tool_call_read"
    assert detail == "view_file"

    # Step 3: Command Tool (run_command)
    cat, detail = detect_step_info(base_dir, conv_id, 3)
    assert cat == "tool_call_command"
    assert detail == "run_command"

    # Step 5: Write Tool (write_to_file)
    cat, detail = detect_step_info(base_dir, conv_id, 5)
    assert cat == "tool_call_write"
    assert detail == "write_to_file"

    # Step 7: Web Tool (search_web)
    cat, detail = detect_step_info(base_dir, conv_id, 7)
    assert cat == "tool_call_web"
    assert detail == "search_web"

    # Step 9: Web Tool (read_url_content)
    cat, detail = detect_step_info(base_dir, conv_id, 9)
    assert cat == "tool_call_web"
    assert detail == "read_url_content"

    # Step 6: Tool Result (step_type = 5 / CODE_ACTION)
    cat, detail = detect_step_info(base_dir, conv_id, 6)
    assert cat == "tool_result"

    # Step 10: Tool Result (step_type = 132 / GENERIC)
    cat, detail = detect_step_info(base_dir, conv_id, 10)
    assert cat == "tool_result"


def test_detect_step_info_payload_fallback(mock_intervention_env):
    """Verifies that detect_step_info correctly classifies tools via DB payload when transcript is missing."""
    base_dir = str(mock_intervention_env["base_dir"])
    conv_id = mock_intervention_env["conv_id"]

    # Temporarily rename transcript logs
    logs_dir = mock_intervention_env["base_dir"] / "brain" / conv_id / ".system_generated" / "logs"
    tf = logs_dir / "transcript_full.jsonl"
    tf_bak = logs_dir / "transcript_full.jsonl.bak"
    tf.rename(tf_bak)

    tc = logs_dir / "transcript.jsonl"
    tc_bak = logs_dir / "transcript.jsonl.bak"
    tc.rename(tc_bak)

    try:
        # Step 7 payload contains b"search_web"
        cat, detail = detect_step_info(base_dir, conv_id, 7)
        assert cat == "tool_call_web"
        assert detail == "search_web"

        # Step 9 payload contains b"read_url_content"
        cat, detail = detect_step_info(base_dir, conv_id, 9)
        assert cat == "tool_call_web"
        assert detail == "read_url_content"
    finally:
        tf_bak.rename(tf)
        tc_bak.rename(tc)


# =========================================================================
# Module 3: Category A & B Intervention Tests (view_file, write_to_file)
# =========================================================================

def test_view_file_intervene(mock_intervention_env):
    base_dir = str(mock_intervention_env["base_dir"])
    conv_id = mock_intervention_env["conv_id"]

    execute_tool_intervene(
        base_dir,
        conv_id,
        target_step=1,
        tool_name="view_file",
        replace_old="DEBUG=False",
        replace_new="DEBUG=True",
        dry_run=False
    )

    # Verify SQLite DB
    conn = sqlite3.connect(str(mock_intervention_env["db_path"]))
    cur = conn.cursor()
    # Step 2 payload updated
    payload = cur.execute("SELECT step_payload FROM steps WHERE idx = 2").fetchone()[0]
    assert b"DEBUG=True" in payload
    # Downstream steps > 2 deleted
    assert cur.execute("SELECT COUNT(*) FROM steps").fetchone()[0] == 3
    assert cur.execute("SELECT MAX(idx) FROM steps").fetchone()[0] == 2
    conn.close()

    # Verify conversation_summaries armed
    s_conn = sqlite3.connect(str(mock_intervention_env["base_dir"] / "conversation_summaries.db"))
    row = s_conn.execute("SELECT step_count, status, not_fully_idle FROM conversation_summaries WHERE conversation_id = ?", (conv_id,)).fetchone()
    assert row[0] == 3
    assert row[1] == "CASCADE_RUN_STATUS_RUNNING"
    assert row[2] == 1
    s_conn.close()


def test_write_to_file_intervene_pure_observation(mock_intervention_env):
    """Verifies that write_to_file intervention modifies DB observation but does not touch disk files."""
    base_dir = str(mock_intervention_env["base_dir"])
    conv_id = mock_intervention_env["conv_id"]

    # Create dummy disk file
    test_file = mock_intervention_env["base_dir"] / "greeting.txt"
    test_file.write_text("Original content created by tool", encoding="utf-8")

    execute_tool_intervene(
        base_dir,
        conv_id,
        target_step=5,
        tool_name="write_to_file",
        replace_text="Error: disk full, could not write greeting.txt",
        dry_run=False
    )

    # 1. Disk file must remain untouched
    assert test_file.exists()
    assert test_file.read_text(encoding="utf-8") == "Original content created by tool"

    # 2. Step 6 payload (step_type = 5) updated
    conn = sqlite3.connect(str(mock_intervention_env["db_path"]))
    payload = conn.execute("SELECT step_payload FROM steps WHERE idx = 6").fetchone()[0]
    assert b"Error: disk full" in payload
    assert conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0] == 7
    conn.close()


# =========================================================================
# Module 4: Category C Command Intervention Tests (run_command)
# =========================================================================

def test_command_intervene_output_sync_and_pruning(mock_intervention_env):
    base_dir = str(mock_intervention_env["base_dir"])
    conv_id = mock_intervention_env["conv_id"]
    steps_dir = mock_intervention_env["steps_dir"]

    # Verify downstream step directories exist before intervention
    assert (steps_dir / "8").is_dir()
    assert (steps_dir / "10").is_dir()

    execute_tool_intervene(
        base_dir,
        conv_id,
        target_step=3,
        tool_name="run_command",
        replace_text="Python 1.0.0 (Legacy)",
        dry_run=False
    )

    # 1. Step 4 output.txt updated
    out_file = steps_dir / "4" / "output.txt"
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == "Python 1.0.0 (Legacy)"

    # 2. Downstream step artifact directories (> 4) purged
    assert not (steps_dir / "8").exists()
    assert not (steps_dir / "10").exists()
    assert not (steps_dir / "12").exists()

    # 3. Transcripts truncated to step 4
    with open(mock_intervention_env["tf_path"], "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    assert len(lines) == 5
    assert lines[-1]["step_index"] == 4
    assert "Python 1.0.0 (Legacy)" in lines[-1]["content"]


# =========================================================================
# Module 5: Category D Web Intervention Tests (search_web, read_url_content)
# =========================================================================

def test_search_web_replace_text(mock_intervention_env):
    base_dir = str(mock_intervention_env["base_dir"])
    conv_id = mock_intervention_env["conv_id"]
    steps_dir = mock_intervention_env["steps_dir"]

    execute_tool_intervene(
        base_dir,
        conv_id,
        target_step=7,
        tool_name="search_web",
        replace_text="The search returned no relevant results.",
        dry_run=False
    )

    # 1. output.txt updated
    out_file = steps_dir / "8" / "output.txt"
    assert out_file.read_text(encoding="utf-8") == "The search returned no relevant results."

    # 2. DB payload updated
    conn = sqlite3.connect(str(mock_intervention_env["db_path"]))
    payload = conn.execute("SELECT step_payload FROM steps WHERE idx = 8").fetchone()[0]
    assert b"The search returned no relevant results." in payload
    conn.close()

    # 3. Downstream step 10, 12 purged
    assert not (steps_dir / "10").exists()
    assert not (steps_dir / "12").exists()


def test_read_url_simulate_error(mock_intervention_env):
    base_dir = str(mock_intervention_env["base_dir"])
    conv_id = mock_intervention_env["conv_id"]
    steps_dir = mock_intervention_env["steps_dir"]

    # Before intervention, content.md exists at steps/10/content.md
    content_md = steps_dir / "10" / "content.md"
    assert content_md.exists()

    execute_tool_intervene(
        base_dir,
        conv_id,
        target_step=9,
        tool_name="read_url_content",
        simulate_error="404",
        dry_run=False
    )

    # 1. content.md MUST be deleted from disk
    assert not content_md.exists()

    # 2. output.txt updated with authentic Antigravity error string
    out_file = steps_dir / "10" / "output.txt"
    assert out_file.exists()
    expected_error = "Encountered error in step execution: Failed to fetch document content at https://example.com/docs: failed to get URL https://example.com/docs: status code 404"
    assert out_file.read_text(encoding="utf-8") == expected_error

    # 3. DB payload updated
    conn = sqlite3.connect(str(mock_intervention_env["db_path"]))
    payload = conn.execute("SELECT step_payload FROM steps WHERE idx = 10").fetchone()[0]
    assert expected_error.encode("utf-8") in payload
    conn.close()

    # 4. Transcript content updated with authentic header and error body
    with open(mock_intervention_env["tf_path"], "r", encoding="utf-8") as f:
        step_10_line = next(json.loads(l) for l in f if json.loads(l)["step_index"] == 10)
    assert "Created At:" in step_10_line["content"]
    assert expected_error in step_10_line["content"]


def test_read_url_replace_web_content_string_and_file(mock_intervention_env, tmp_path):
    base_dir = str(mock_intervention_env["base_dir"])
    conv_id = mock_intervention_env["conv_id"]
    steps_dir = mock_intervention_env["steps_dir"]
    content_md = steps_dir / "10" / "content.md"

    # 1. Test replacement via raw string
    execute_tool_intervene(
        base_dir,
        conv_id,
        target_step=9,
        tool_name="read_url_content",
        replace_web_content="# Injected Fake Documentation\nManipulated body.",
        dry_run=False
    )

    assert content_md.exists()
    assert content_md.read_text(encoding="utf-8") == "# Injected Fake Documentation\nManipulated body."

    # Pointer metadata in output.txt must remain intact
    out_file = steps_dir / "10" / "output.txt"
    assert "The full content of the article at https://example.com/docs has been saved to:" in out_file.read_text(encoding="utf-8")

    # 2. Test replacement via external file path
    replacement_file = tmp_path / "custom_doc.md"
    replacement_file.write_text("# From External File\nFile content.", encoding="utf-8")

    execute_tool_intervene(
        base_dir,
        conv_id,
        target_step=9,
        tool_name="read_url_content",
        replace_web_content=str(replacement_file),
        dry_run=False
    )
    assert content_md.read_text(encoding="utf-8") == "# From External File\nFile content."


# =========================================================================
# Module 6: Step Output Directory Synchronization (sync_or_prune_step_dirs)
# =========================================================================

def test_sync_or_prune_step_dirs(tmp_path):
    steps_dir = tmp_path / "steps"
    (steps_dir / "2").mkdir(parents=True)
    (steps_dir / "2" / "output.txt").write_text("Original output 2", encoding="utf-8")
    (steps_dir / "2" / "content.md").write_text("Original content 2", encoding="utf-8")

    (steps_dir / "5").mkdir(parents=True)
    (steps_dir / "5" / "output.txt").write_text("Original output 5", encoding="utf-8")

    # 1. Prune downstream directories (> 2) and update output without touching content.md
    purged, modified, deleted = sync_or_prune_step_dirs(
        str(steps_dir),
        cutoff_step=2,
        modify_step=2,
        new_str="Updated output 2",
        dry_run=False
    )
    assert purged == 1
    assert modified == 1
    assert deleted == 0
    assert not (steps_dir / "5").exists()
    assert (steps_dir / "2" / "output.txt").read_text(encoding="utf-8") == "Updated output 2"
    assert (steps_dir / "2" / "content.md").read_text(encoding="utf-8") == "Original content 2"

    # 2. Delete content.md on delete_web_content flag
    purged2, modified2, deleted2 = sync_or_prune_step_dirs(
        str(steps_dir),
        cutoff_step=2,
        modify_step=2,
        delete_web_content=True,
        dry_run=False
    )
    assert deleted2 == 1
    assert not (steps_dir / "2" / "content.md").exists()


# =========================================================================
# Module 7: Kickstart Cleanup & Index Re-alignment (execute_kickstart_cleanup)
# =========================================================================

def test_kickstart_cleanup_shifts_indices(mock_intervention_env):
    base_dir = str(mock_intervention_env["base_dir"])
    conv_id = mock_intervention_env["conv_id"]

    # Initial state: Step 13 is kickstart, Step 14 is model response (total 15 steps)
    conn = sqlite3.connect(str(mock_intervention_env["db_path"]))
    assert conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0] == 15
    conn.close()

    execute_kickstart_cleanup(
        base_dir,
        conv_id,
        token="__AGY_KICKSTART__",
        dry_run=False
    )

    # 1. Step 13 deleted, Step 14 shifted to index 13
    conn = sqlite3.connect(str(mock_intervention_env["db_path"]))
    rows = conn.execute("SELECT idx, step_type FROM steps ORDER BY idx").fetchall()
    assert len(rows) == 14
    assert rows[-1] == (13, 15)  # Former Step 14 is now Step 13
    conn.close()

    # 2. History cleaned
    with open(mock_intervention_env["hist_file"], "r", encoding="utf-8") as f:
        hist_lines = [json.loads(l) for l in f]
    assert not any("__AGY_KICKSTART__" in h.get("display", "") for h in hist_lines)
    assert any("other conversation prompt" in h.get("display", "") for h in hist_lines)


# =========================================================================
# Module 8: CLI Option Validation & Negative Tests (main())
# =========================================================================

def test_cli_validation_rules(mock_intervention_env, monkeypatch, capsys):
    base_dir = str(mock_intervention_env["base_dir"])
    conv_id = mock_intervention_env["conv_id"]

    def run_cli(*args):
        test_argv = ["Antigravity_Rewind.py", "--dir", base_dir, "--conv", conv_id, "--dry-run"] + list(args)
        monkeypatch.setattr(sys, "argv", test_argv)
        with pytest.raises(SystemExit) as exc_info:
            main()
        out, err = capsys.readouterr()
        return exc_info.value.code, out, err

    # 1. Disallow interventions on Prompt (Step 0)
    code, out, err = run_cli("0", "--simulate-error")
    assert code == 1
    assert "Intervention options" in err and "Prompt" in err

    # 2. Disallow --simulate-error on view_file (Step 1)
    code, out, err = run_cli("1", "--simulate-error")
    assert code == 1
    assert "--simulate-error is only supported for 'read_url_content' and command tool steps" in err

    # 3. Disallow --simulate-error on search_web (Step 7)
    code, out, err = run_cli("7", "--simulate-error")
    assert code == 1
    assert "--simulate-error is only supported for 'read_url_content' and command tool steps" in err

    # 4. Disallow --replace-web-content on search_web (Step 7)
    code, out, err = run_cli("7", "--replace-web-content", "mock content")
    assert code == 1
    assert "--replace-web-content is only supported for 'read_url_content'" in err

    # 5. Disallow both --simulate-error and --replace-web-content concurrently (Step 9)
    code, out, err = run_cli("9", "--simulate-error", "--replace-web-content", "mock content")
    assert code == 1
    assert "Cannot specify both --simulate-error and --replace-web-content simultaneously" in err

    # 6. Disallow tool intervention without any replacement content (Step 9)
    code, out, err = run_cli("9")
    assert code == 1
    assert "requires intervention content" in err
