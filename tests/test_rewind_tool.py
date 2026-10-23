import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pytest

try:
    import rewinding.Antigravity_Rewind as rewind_mod
except ImportError:
    import rewinding.rewind_tool as rewind_mod

resolve_base_dir = rewind_mod.resolve_base_dir
resolve_dest_base_dir = rewind_mod.resolve_dest_base_dir
get_conversation_id = rewind_mod.get_conversation_id
get_conversation_info = rewind_mod.get_conversation_info
clean_preview = rewind_mod.clean_preview
format_event = rewind_mod.format_event
get_step_items = rewind_mod.get_step_items
list_events = rewind_mod.list_events
search_events = rewind_mod.search_events
count_or_truncate_jsonl = rewind_mod.count_or_truncate_jsonl
count_or_truncate_chunks = rewind_mod.count_or_truncate_chunks
count_or_filter_history = rewind_mod.count_or_filter_history
execute_rewind = rewind_mod.execute_rewind
sync_pipeline = rewind_mod.sync_pipeline


@pytest.fixture
def mock_rewind_env(tmp_path):
    """
    Constructs a realistic, isolated Antigravity .gemini environment
    with a multi-turn conversation (steps 0 to 7), SQLite database,
    summaries, proto cache, full & compact transcripts, chunks,
    multi-session history, and presence locks.
    """
    gemini_root = tmp_path / ".gemini"
    base_dir = gemini_root / "antigravity-cli"
    base_dir.mkdir(parents=True)

    conv_id = "test-conv-uuid-1234"

    # 1. conversations/<conv_id>.db
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

    # Insert 8 steps (0 to 7)
    # Even steps (0, 2, 4, 6) = USER_INPUT (14)
    # Odd steps (1, 3, 5, 7) = MODEL_RESPONSE (15)
    for idx in range(8):
        stype = 14 if idx % 2 == 0 else 15
        cur.execute("INSERT INTO steps (idx, step_type, status, step_payload) VALUES (?, ?, ?, ?)",
                    (idx, stype, 1, f"payload-{idx}".encode()))
        cur.execute("INSERT INTO parent_references (idx, data) VALUES (?, ?)", (idx, b"parent-ref"))
        cur.execute("INSERT INTO battle_mode_infos (idx, data) VALUES (?, ?)", (idx, b"battle-info"))

    # 4 model turns (idx 0 to 3)
    for m_idx in range(4):
        cur.execute("INSERT INTO gen_metadata (idx, data) VALUES (?, ?)", (m_idx, b"gen-meta"))
        cur.execute("INSERT INTO executor_metadata (idx, data) VALUES (?, ?)", (m_idx, b"exec-meta"))

    conn.commit()
    conn.close()

    # 2. conversation_summaries.db & proto cache
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
            raw_summary BLOB
        )
    """)
    s_cur.execute(
        "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?)",
        (conv_id, "Mock Title", "Mock Preview", 8, 6, b"mock-raw-summary")
    )
    s_conn.commit()
    s_conn.close()

    proto_cache = base_dir / "jetbox_summaries_proto.pb"
    proto_cache.write_bytes(b"mock-protobuf-cache")

    # 3. presence lock
    presence_dir = base_dir / "presence"
    presence_dir.mkdir()
    lock_file = presence_dir / f"{conv_id}.lock"
    lock_file.write_text("", encoding="utf-8")

    # 4. brain logs and chunks
    logs_dir = base_dir / "brain" / conv_id / ".system_generated" / "logs"
    chunks_tf_dir = logs_dir / "chunks" / "transcript_full"
    chunks_tc_dir = logs_dir / "chunks" / "transcript"
    chunks_tf_dir.mkdir(parents=True)
    chunks_tc_dir.mkdir(parents=True)

    timestamps = [
        "2026-10-16T08:30:00Z",
        "2026-10-16T08:30:05Z",
        "2026-10-16T08:31:00Z",
        "2026-10-16T08:31:10Z",
        "2026-10-16T08:32:00Z",
        "2026-10-16T08:32:05Z",
        "2026-10-16T08:33:00Z",
        "2026-10-16T08:33:05Z",
    ]

    steps_data = []
    for idx in range(8):
        if idx % 2 == 0:
            prompt_text = f"Prompt query number {idx}"
            s = {
                "step_index": idx,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": timestamps[idx],
                "content": f"<USER_REQUEST>\n{prompt_text}\n</USER_REQUEST>",
                "status": "DONE",
            }
        else:
            s = {
                "step_index": idx,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": timestamps[idx],
                "content": f"Response to turn {idx - 1}",
                "status": "DONE",
            }
            if idx == 3:
                s["tool_calls"] = [{"name": "run_command", "args": {"CommandLine": "echo hello", "Cwd": "/tmp"}}]
                s["thinking"] = "Need to test shell command"
        steps_data.append(s)

    # Write transcript_full.jsonl and transcript.jsonl
    tf_path = logs_dir / "transcript_full.jsonl"
    tc_path = logs_dir / "transcript.jsonl"
    with open(tf_path, "w", encoding="utf-8") as f_full, open(tc_path, "w", encoding="utf-8") as f_comp:
        for s in steps_data:
            line = json.dumps(s) + "\n"
            f_full.write(line)
            f_comp.write(line)

    # Write chunk files: chunk 0 (steps 0-3), chunk 1 (steps 4-7)
    chunk0_tf = chunks_tf_dir / "00000000.jsonl"
    chunk1_tf = chunks_tf_dir / "00000001.jsonl"
    chunk0_tc = chunks_tc_dir / "00000000.jsonl"
    chunk1_tc = chunks_tc_dir / "00000001.jsonl"

    with open(chunk0_tf, "w", encoding="utf-8") as c0_f, open(chunk0_tc, "w", encoding="utf-8") as c0_c:
        for s in steps_data[:4]:
            l = json.dumps(s) + "\n"
            c0_f.write(l)
            c0_c.write(l)

    with open(chunk1_tf, "w", encoding="utf-8") as c1_f, open(chunk1_tc, "w", encoding="utf-8") as c1_c:
        for s in steps_data[4:]:
            l = json.dumps(s) + "\n"
            c1_f.write(l)
            c1_c.write(l)

    # 5. history.jsonl
    hist_path = base_dir / "history.jsonl"
    hist_entries = [
        # Unrelated session entries
        {"conversationId": "other-session-999", "display": "other user task", "timestamp": 1700000000000},
        {"display": "global command", "timestamp": 1700000001000},
        # Target session entries
        {"conversationId": conv_id, "display": "Prompt query number 0", "timestamp": 1792139400000},
        {"conversationId": conv_id, "display": "Prompt query number 2", "timestamp": 1792139460000},
        {"conversationId": conv_id, "display": "Prompt query number 4", "timestamp": 1792139520000},
        {"conversationId": conv_id, "display": "Prompt query number 6", "timestamp": 1792139580000},
        {"conversationId": conv_id, "display": "/exit", "timestamp": 1792139590000, "type": "slash_command"},
    ]
    with open(hist_path, "w", encoding="utf-8") as f:
        for h in hist_entries:
            f.write(json.dumps(h) + "\n")

    return {
        "gemini_root": gemini_root,
        "base_dir": base_dir,
        "conv_id": conv_id,
        "db_path": db_path,
        "sum_db": sum_db,
        "proto_cache": proto_cache,
        "presence_lock": lock_file,
        "tf_path": tf_path,
        "tc_path": tc_path,
        "chunks_tf_dir": chunks_tf_dir,
        "chunks_tc_dir": chunks_tc_dir,
        "hist_path": hist_path,
    }


# ==============================================================================
# 1. Path Resolution Tests
# ==============================================================================

def test_resolve_base_dir(mock_rewind_env):
    gemini_root = str(mock_rewind_env["gemini_root"])
    base_dir = str(mock_rewind_env["base_dir"])

    # Resolving from parent .gemini returns .gemini/antigravity-cli
    assert resolve_base_dir(gemini_root) == base_dir
    # Resolving directly from antigravity-cli returns antigravity-cli
    assert resolve_base_dir(base_dir) == base_dir


def test_resolve_dest_base_dir(mock_rewind_env, tmp_path):
    gemini_root = str(mock_rewind_env["gemini_root"])
    dest_gemini = str(tmp_path / "cloned_gemini")

    # Cloned destination preserves the antigravity-cli subpath
    dest_base = resolve_dest_base_dir(gemini_root, dest_gemini)
    assert dest_base == os.path.join(dest_gemini, "antigravity-cli")

    # In-place destination returns current base dir
    assert resolve_dest_base_dir(gemini_root, gemini_root) == str(mock_rewind_env["base_dir"])


def test_get_conversation_id(mock_rewind_env):
    base_dir = str(mock_rewind_env["base_dir"])
    conv_id = mock_rewind_env["conv_id"]

    # Auto-detects single conversation
    assert get_conversation_id(base_dir) == conv_id
    # Explicit conversation name
    assert get_conversation_id(base_dir, conv_id) == conv_id
    assert get_conversation_id(base_dir, f"{conv_id}.db") == conv_id

    # Non-existent conversation raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        get_conversation_id(base_dir, "non-existent-uuid")


def test_get_conversation_info_and_multiple_listing(mock_rewind_env, capsys):
    base_dir = str(mock_rewind_env["base_dir"])
    conv_id = mock_rewind_env["conv_id"]

    # 1. Conv 1 has summary in conversation_summaries.db (mock-raw-summary)
    # Let's update conversation_summaries to have a title
    with sqlite3.connect(str(mock_rewind_env["sum_db"])) as conn:
        conn.execute("UPDATE conversation_summaries SET title = 'Math Calculations' WHERE conversation_id = ?", (conv_id,))

    count, summary = get_conversation_info(base_dir, conv_id)
    assert count == 8
    assert summary == "Math Calculations"

    # 2. Add second conversation with NO summary, but has prompt in transcript
    conv2_id = "test-conv-fallback-prompt"
    conv2_db = Path(base_dir) / "conversations" / f"{conv2_id}.db"
    with sqlite3.connect(str(conv2_db)) as conn:
        conn.execute("CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER)")
        conn.execute("INSERT INTO steps VALUES (0, 14)")
    
    conv2_logs = Path(base_dir) / "brain" / conv2_id / ".system_generated" / "logs"
    conv2_logs.mkdir(parents=True)
    with open(conv2_logs / "transcript_full.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"source": "USER_EXPLICIT", "content": "<USER_REQUEST>\nAnalyze python logs\n</USER_REQUEST>"}) + "\n")

    count2, summary2 = get_conversation_info(base_dir, conv2_id)
    assert count2 == 1
    assert summary2 == "Analyze python logs"

    # 3. Add third conversation with NO prompt, but has slash command in history.jsonl
    conv3_id = "test-conv-fallback-slash"
    conv3_db = Path(base_dir) / "conversations" / f"{conv3_id}.db"
    with sqlite3.connect(str(conv3_db)) as conn:
        conn.execute("CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER)")
    with open(mock_rewind_env["hist_path"], "a", encoding="utf-8") as f:
        f.write(json.dumps({"conversationId": conv3_id, "display": "/exit", "type": "slash_command"}) + "\n")

    count3, summary3 = get_conversation_info(base_dir, conv3_id)
    assert count3 == 0
    assert summary3 == "/exit"

    # 4. Add fourth conversation with NOTHING
    conv4_id = "test-conv-empty"
    conv4_db = Path(base_dir) / "conversations" / f"{conv4_id}.db"
    with sqlite3.connect(str(conv4_db)) as conn:
        conn.execute("CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER)")

    count4, summary4 = get_conversation_info(base_dir, conv4_id)
    assert count4 == 0
    assert summary4 == ""

    # Test listing multiple conversations raises SystemExit and prints detailed summaries
    with pytest.raises(SystemExit):
        get_conversation_id(base_dir)

    out = capsys.readouterr().out
    assert "Multiple conversations found. Please specify one with --conv:" in out
    assert f"- {conv_id} (8 steps): Math Calculations" in out
    assert f"- {conv2_id} (1 step): Analyze python logs" in out
    assert f"- {conv3_id} (0 steps): /exit" in out
    assert f"- {conv4_id} (0 steps)" in out



# ==============================================================================
# 2. Event Formatting and Tree Grouping Tests
# ==============================================================================

def test_clean_preview():
    raw_prompt = (
        "<USER_REQUEST>\nCalculate 4 + 4\n</USER_REQUEST>\n"
        "<ADDITIONAL_METADATA>\ntime=now\n</ADDITIONAL_METADATA>\n"
        "<USER_SETTINGS_CHANGE>\nmodel=flash\n</USER_SETTINGS_CHANGE>"
    )
    assert clean_preview(raw_prompt) == "Calculate 4 + 4"
    assert clean_preview("") == ""
    assert clean_preview("Short text") == "Short text"


def test_format_event():
    # Prompt event
    ev_prompt = {"event_type": "prompt", "text": "Hello world"}
    assert format_event(ev_prompt) == ("PROMPT", "Hello world")

    # Agent message event
    ev_resp = {"event_type": "agent_message", "text": "I can help with that"}
    assert format_event(ev_resp) == ("RESPONSE", "I can help with that")

    # Tool call event (run_command)
    ev_cmd = {
        "event_type": "tool_call",
        "tool_name": "run_command",
        "command": "python script.py",
        "tool_input": {"Cwd": "/project"},
    }
    label, preview = format_event(ev_cmd)
    assert label == "TOOL CALL: run_command"
    assert "$ python script.py (cwd: /project)" in preview

    # Tool call event (view_file)
    ev_file = {
        "event_type": "tool_call",
        "tool_name": "view_file",
        "target_paths": ["/home/user/app.py"],
    }
    label, preview = format_event(ev_file)
    assert label == "TOOL CALL: view_file"
    assert "/home/user/app.py" in preview

    # Tool result event
    ev_res = {"event_type": "tool_result", "tool_name": "run_command", "tool_output": "success output", "status": "DONE"}
    assert format_event(ev_res) == ("TOOL RESULT: run_command", "success output")

    # Slash command event
    ev_slash = {"event_type": "slash_command", "text": "/exit"}
    assert format_event(ev_slash) == ("SLASH CMD", "/exit")


def test_get_step_items():
    step_events = [
        {"event_type": "tool_call", "tool_name": "run_command", "command": "ls -l", "thinking": "Investigating directory"},
        {"event_type": "tool_result", "tool_name": "run_command", "text": "file.txt"},
        {"event_type": "agent_message", "text": "Found file.txt"},
    ]
    items = get_step_items(step_events)
    labels = [it[0] for it in items]
    assert labels == ["THINKING", "TOOL CALL: run_command", "TOOL RESULT: run_command", "RESPONSE"]
    assert items[0][1] == "Investigating directory"


def test_list_events(capsys):
    events = [
        {"step_index": 0, "event_type": "prompt", "text": "Hello"},
        {"step_index": 1, "event_type": "tool_call", "tool_name": "run_command", "command": "echo 1", "thinking": "Let me test"},
        {"step_index": 1, "event_type": "agent_message", "text": "Result is 1"},
        {"step_index": None, "event_type": "slash_command", "text": "/exit"},
    ]
    list_events(events, "conv-test-id")
    captured = capsys.readouterr().out

    assert "Conversation steps (conv-test-id):" in captured
    assert "Step  0 [PROMPT]: Hello" in captured
    assert "Step  1:" in captured
    assert "├─ [THINKING]: Let me test" in captured
    assert "└─ [RESPONSE]: Result is 1" in captured
    assert "[Timeline] [SLASH CMD] (non-rewindable): /exit" in captured


# ==============================================================================
# 3. Search Events Tests
# ==============================================================================

def test_search_events(capsys):
    events = [
        {"step_index": 0, "event_type": "prompt", "text": "Calculate 4 + 4"},
        {"step_index": 1, "event_type": "agent_message", "text": "The answer is 8"},
        {"step_index": 2, "event_type": "tool_call", "tool_name": "run_command", "command": "pytest -v", "tool_input": {}},
        {"step_index": None, "event_type": "slash_command", "text": "/exit"},
    ]

    # Match in prompt
    search_events(events, "4 + 4")
    out = capsys.readouterr().out
    assert "Found 1 matching event(s)" in out
    assert "Step  0 [PROMPT]: Calculate 4 + 4" in out
    assert "python rewind_tool.py 0" in out

    # Match in command
    search_events(events, "pytest")
    out = capsys.readouterr().out
    assert "Step  2 [TOOL CALL: run_command]" in out
    assert "python rewind_tool.py 2" in out

    # Match in non-rewindable event
    search_events(events, "exit")
    out = capsys.readouterr().out
    assert "[Timeline] [SLASH CMD] (non-rewindable): /exit" in out
    assert "None of the matching events have a rewindable step index." in out

    # No match
    search_events(events, "nonexistent-query")
    out = capsys.readouterr().out
    assert "No steps or events found matching query: 'nonexistent-query'" in out


# ==============================================================================
# 4. Truncation and Filtering Unit Tests
# ==============================================================================

def test_count_or_truncate_jsonl(tmp_path):
    jsonl_path = tmp_path / "test.jsonl"
    lines = [
        {"step_index": 0, "text": "step 0"},
        {"step_index": 1, "text": "step 1"},
        {"step_index": 2, "text": "step 2"},
        {"step_index": 3, "text": "step 3"},
    ]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")

    # Dry-run: returns count without modifying file
    removed = count_or_truncate_jsonl(str(jsonl_path), target_step=2, dry_run=True)
    assert removed == 2
    with open(jsonl_path, "r", encoding="utf-8") as f:
        assert len(f.readlines()) == 4

    # Live run: truncates file to steps < 2
    removed = count_or_truncate_jsonl(str(jsonl_path), target_step=2, dry_run=False)
    assert removed == 2
    with open(jsonl_path, "r", encoding="utf-8") as f:
        remaining = [json.loads(line) for line in f]
    assert len(remaining) == 2
    assert [r["step_index"] for r in remaining] == [0, 1]


def test_count_or_truncate_chunks(tmp_path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunk0 = chunks_dir / "00000000.jsonl"
    chunk1 = chunks_dir / "00000001.jsonl"

    with open(chunk0, "w", encoding="utf-8") as f:
        f.write(json.dumps({"step_index": 0}) + "\n")
        f.write(json.dumps({"step_index": 1}) + "\n")

    with open(chunk1, "w", encoding="utf-8") as f:
        f.write(json.dumps({"step_index": 2}) + "\n")
        f.write(json.dumps({"step_index": 3}) + "\n")

    # Rewind to step 2: chunk0 has 2 kept lines, chunk1 becomes empty and is deleted
    rem_lines, del_files = count_or_truncate_chunks(str(chunks_dir), target_step=2, dry_run=False)
    assert rem_lines == 2
    assert del_files == 1
    assert chunk0.exists()
    assert not chunk1.exists()


def test_count_or_filter_history(tmp_path):
    hist_path = tmp_path / "history.jsonl"
    conv_id = "test-conv"
    cutoff_ms = 10000

    entries = [
        {"conversationId": "other-conv", "display": "other task", "timestamp": 15000},
        {"conversationId": conv_id, "display": "old prompt", "timestamp": 5000},
        {"conversationId": conv_id, "display": "new prompt", "timestamp": 12000},
        {"conversationId": conv_id, "display": "/exit", "timestamp": 13000, "type": "slash_command"},
    ]
    with open(hist_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    removed = count_or_filter_history(str(hist_path), conv_id, cutoff_ms=cutoff_ms, rewound_prompts={"new prompt"}, dry_run=False)
    assert removed == 2

    with open(hist_path, "r", encoding="utf-8") as f:
        remaining = [json.loads(l) for l in f]

    assert len(remaining) == 2
    # Other conversation preserved
    assert remaining[0]["conversationId"] == "other-conv"
    # Old prompt of this conversation preserved
    assert remaining[1]["display"] == "old prompt"


# ==============================================================================
# 5. Full Rewind Execution Tests (In-Place, Separate Dest, Dry-Run)
# ==============================================================================

def test_execute_rewind_in_place(mock_rewind_env):
    base_dir = str(mock_rewind_env["base_dir"])
    conv_id = mock_rewind_env["conv_id"]

    # Rewind to Step 2
    execute_rewind(base_dir, conv_id, target_step=2, dry_run=False)

    # 1. DB checks
    conn = sqlite3.connect(str(mock_rewind_env["db_path"]))
    cur = conn.cursor()
    remaining_steps = cur.execute("SELECT idx, step_type FROM steps ORDER BY idx").fetchall()
    assert len(remaining_steps) == 2
    assert [r[0] for r in remaining_steps] == [0, 1]

    # gen_metadata trimmed to 1 model turn
    assert cur.execute("SELECT COUNT(*) FROM gen_metadata").fetchone()[0] == 1
    assert cur.execute("SELECT COUNT(*) FROM executor_metadata").fetchone()[0] == 1

    # parent_references & battle_mode_infos trimmed
    assert cur.execute("SELECT COUNT(*) FROM parent_references").fetchone()[0] == 2
    assert cur.execute("SELECT COUNT(*) FROM battle_mode_infos").fetchone()[0] == 2
    conn.close()

    # 2. Summary DB check
    s_conn = sqlite3.connect(str(mock_rewind_env["sum_db"]))
    s_cur = s_conn.cursor()
    row = s_cur.execute("SELECT step_count, last_user_input_step_index, raw_summary FROM conversation_summaries WHERE conversation_id = ?", (conv_id,)).fetchone()
    assert row[0] == 2
    assert row[1] == 0
    assert row[2] is None
    s_conn.close()

    # 3. Proto cache and presence lock removed
    assert not mock_rewind_env["proto_cache"].exists()
    assert not mock_rewind_env["presence_lock"].exists()

    # 4. Transcripts truncated
    with open(mock_rewind_env["tf_path"], "r", encoding="utf-8") as f:
        tf_lines = [json.loads(l) for l in f]
    assert len(tf_lines) == 2
    assert [s["step_index"] for s in tf_lines] == [0, 1]

    with open(mock_rewind_env["tc_path"], "r", encoding="utf-8") as f:
        tc_lines = [json.loads(l) for l in f]
    assert len(tc_lines) == 2

    # 5. Chunks: chunk 0 truncated, chunk 1 (empty) deleted
    chunk0 = mock_rewind_env["chunks_tf_dir"] / "00000000.jsonl"
    chunk1 = mock_rewind_env["chunks_tf_dir"] / "00000001.jsonl"
    assert chunk0.exists()
    assert not chunk1.exists()

    # 6. History: other conversation preserved, steps >= 2 removed
    with open(mock_rewind_env["hist_path"], "r", encoding="utf-8") as f:
        hist_lines = [json.loads(l) for l in f]
    assert any(h.get("conversationId") == "other-session-999" for h in hist_lines)
    assert any(h.get("display") == "global command" for h in hist_lines)
    assert any(h.get("display") == "Prompt query number 0" for h in hist_lines)
    assert not any(h.get("display") == "Prompt query number 2" for h in hist_lines)
    assert not any(h.get("display") == "Prompt query number 4" for h in hist_lines)
    assert not any(h.get("display") == "/exit" for h in hist_lines)


def test_execute_rewind_separate_dest(mock_rewind_env, tmp_path):
    source_gemini = str(mock_rewind_env["gemini_root"])
    dest_gemini = str(tmp_path / "separate_rewound_gemini")
    conv_id = mock_rewind_env["conv_id"]

    # Clone source to dest
    shutil.copytree(source_gemini, dest_gemini)

    dest_base = resolve_dest_base_dir(source_gemini, dest_gemini)

    # Rewind the destination to Step 4
    execute_rewind(dest_base, conv_id, target_step=4, dry_run=False, source_base_dir=str(mock_rewind_env["base_dir"]))

    # Verify source environment is completely untouched (all 8 steps remain)
    conn_src = sqlite3.connect(str(mock_rewind_env["db_path"]))
    assert conn_src.execute("SELECT COUNT(*) FROM steps").fetchone()[0] == 8
    conn_src.close()
    assert mock_rewind_env["proto_cache"].exists()
    with open(mock_rewind_env["tf_path"], "r", encoding="utf-8") as f:
        assert len(f.readlines()) == 8

    # Verify destination environment has been rewound to Step 4
    dest_db = os.path.join(dest_base, "conversations", f"{conv_id}.db")
    conn_dest = sqlite3.connect(dest_db)
    assert conn_dest.execute("SELECT COUNT(*) FROM steps").fetchone()[0] == 4
    conn_dest.close()

    dest_tf = os.path.join(dest_base, "brain", conv_id, ".system_generated", "logs", "transcript_full.jsonl")
    with open(dest_tf, "r", encoding="utf-8") as f:
        assert len(f.readlines()) == 4


def test_execute_rewind_dry_run(mock_rewind_env, capsys):
    base_dir = str(mock_rewind_env["base_dir"])
    conv_id = mock_rewind_env["conv_id"]

    execute_rewind(base_dir, conv_id, target_step=2, dry_run=True)
    out = capsys.readouterr().out

    assert "[DRY-RUN] Rewind target: delete steps >= 2" in out
    assert "[DRY-RUN] Steps remaining: 2" in out
    assert "[DRY-RUN] Steps to be deleted from DB: 6" in out
    assert "[DRY-RUN] Lines to truncate in transcript_full.jsonl: 6" in out
    assert "[DRY-RUN] Entries to filter from history.jsonl: 4" in out

    # Verify nothing was modified on disk
    conn = sqlite3.connect(str(mock_rewind_env["db_path"]))
    assert conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0] == 8
    conn.close()

    assert mock_rewind_env["proto_cache"].exists()
    assert mock_rewind_env["presence_lock"].exists()
    with open(mock_rewind_env["tf_path"], "r", encoding="utf-8") as f:
        assert len(f.readlines()) == 8


def test_execute_rewind_real_sample_dataset(tmp_path):
    sample_dir = Path("/home/kimseungsu/temp/rewind/simple rewind/.gemini")
    if not sample_dir.is_dir():
        pytest.skip(f"Real sample dataset not found at {sample_dir}")

    # Copy real sample dataset into isolated tmp_path
    cloned_gemini = tmp_path / ".gemini"
    shutil.copytree(str(sample_dir), str(cloned_gemini))

    conv_id = "4fcc6ec6-1a0f-4800-8d15-6393dffc703c"
    base_dir = resolve_base_dir(str(cloned_gemini))

    # Verify initial state has 8 steps
    db_path = os.path.join(base_dir, "conversations", f"{conv_id}.db")
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0] == 8

    # Rewind to step 2
    execute_rewind(base_dir, conv_id, target_step=2, dry_run=False)

    # Verify DB has 2 steps remaining
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0] == 2
        assert [r[0] for r in conn.execute("SELECT idx FROM steps ORDER BY idx").fetchall()] == [0, 1]

    # Verify full transcript truncated to 2 lines
    tf_path = os.path.join(base_dir, "brain", conv_id, ".system_generated", "logs", "transcript_full.jsonl")
    with open(tf_path, "r", encoding="utf-8") as f:
        tf_steps = [json.loads(line) for line in f]
    assert len(tf_steps) == 2
    assert [s["step_index"] for s in tf_steps] == [0, 1]

    # Verify history.jsonl filtered
    hist_path = os.path.join(base_dir, "history.jsonl")
    with open(hist_path, "r", encoding="utf-8") as f:
        remaining_history = [json.loads(line) for line in f]
    assert not any(h.get("display") in ("what is 2+2?", "what is 4+4?", "what comes next?", "/exit") and h.get("conversationId") == conv_id for h in remaining_history)
    assert any(h.get("display") == "what is 1+1?" for h in remaining_history)

