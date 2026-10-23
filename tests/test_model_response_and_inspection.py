import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
import pytest

import rewinding.Antigravity_Rewind as rewind_mod

detect_step_info = rewind_mod.detect_step_info
execute_rewind = rewind_mod.execute_rewind
show_step = rewind_mod.show_step
print_standout_box = rewind_mod.print_standout_box
main = rewind_mod.main


@pytest.fixture
def mock_model_response_session(tmp_path):
    """
    Constructs an isolated, multi-turn Antigravity session covering:
      - Step 0: User Prompt (USER_INPUT = 14)
      - Step 1: Model Response with Tool Call run_command (MODEL_RESPONSE = 15)
      - Step 2: Tool Result run_command (step_type = 132, with step artifact directory)
      - Step 3: Model Response with Tool Call view_file (MODEL_RESPONSE = 15)
      - Step 4: Tool Result view_file (step_type = 132)
      - Step 5: Pure Model Response (text only, MODEL_RESPONSE = 15, no tool call)
      - Step 6: User Prompt (USER_INPUT = 14)
      - Step 7: Model Response with Tool Call ask_question (MODEL_RESPONSE = 15)
      - Step 8: Tool Result ask_question (step_type = 132)
      - Step 9: Pure Model Response (text only, MODEL_RESPONSE = 15)
    """
    gemini_root = tmp_path / ".gemini"
    base_dir = gemini_root / "antigravity-cli"
    base_dir.mkdir(parents=True)

    conv_id = "test-model-response-conv-5555"

    # 1. Database setup
    convs_dir = base_dir / "conversations"
    convs_dir.mkdir()
    db_path = convs_dir / f"{conv_id}.db"

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, status INTEGER, metadata BLOB, step_payload BLOB)")
    cur.execute("CREATE TABLE gen_metadata (idx INTEGER PRIMARY KEY, data BLOB)")
    cur.execute("CREATE TABLE executor_metadata (idx INTEGER PRIMARY KEY, data BLOB)")
    cur.execute("CREATE TABLE parent_references (idx INTEGER PRIMARY KEY, data BLOB)")
    cur.execute("CREATE TABLE battle_mode_infos (idx INTEGER PRIMARY KEY, data BLOB)")

    def make_proto_payload(text: str) -> bytes:
        b = text.encode("utf-8")
        sub = rewind_mod.serialize_fields([(4, 2, b)])
        return rewind_mod.serialize_fields([(14, 2, sub)])

    steps_data = [
        (0, 14, 3, None, b"User prompt: Inspect application"),
        (1, 15, 3, None, b"run_command ls -la"),
        (2, 132, 3, None, make_proto_payload("total 4\n-rw-r--r-- app.py")),
        (3, 15, 3, None, b"view_file /test/app.py"),
        (4, 132, 3, None, make_proto_payload("print('hello')")),
        (5, 15, 3, None, b"I have analyzed app.py and found no issues."),
        (6, 14, 3, None, b"Now ask me a question to confirm deployment."),
        (7, 15, 3, None, b"ask_question"),
        (8, 132, 3, None, make_proto_payload("yes")),
        (9, 15, 3, None, b"Deployment confirmed. Proceeding now."),
    ]

    for idx, stype, status, meta, payload in steps_data:
        cur.execute("INSERT INTO steps VALUES (?, ?, ?, ?, ?)", (idx, stype, status, meta, payload))
        cur.execute("INSERT INTO parent_references VALUES (?, ?)", (idx, b"pref"))
        cur.execute("INSERT INTO battle_mode_infos VALUES (?, ?)", (idx, b"binfo"))

    for m_idx in range(5):
        cur.execute("INSERT INTO gen_metadata VALUES (?, ?)", (m_idx, b"gen-meta"))
        cur.execute("INSERT INTO executor_metadata VALUES (?, ?)", (m_idx, b"exec-meta"))

    conn.commit()
    conn.close()

    # 2. conversation_summaries.db
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
        (conv_id, "Model Response Test", "Session preview", 10, 6, "CASCADE_RUN_STATUS_IDLE", 0, b"raw-sum")
    )
    s_conn.commit()
    s_conn.close()

    # 3. Transcripts, chunks, steps, and messages directories
    brain_dir = base_dir / "brain" / conv_id
    sys_gen = brain_dir / ".system_generated"
    logs_dir = sys_gen / "logs"
    chunks_full = logs_dir / "chunks" / "transcript_full"
    chunks_comp = logs_dir / "chunks" / "transcript"
    steps_dir = sys_gen / "steps"
    tasks_dir = sys_gen / "tasks"
    messages_dir = sys_gen / "messages"

    for d in [chunks_full, chunks_comp, steps_dir, tasks_dir, messages_dir]:
        d.mkdir(parents=True)

    # Step 2 artifact directory
    (steps_dir / "2").mkdir()
    (steps_dir / "2" / "output.txt").write_text("total 4\n-rw-r--r-- app.py\n", encoding="utf-8")

    tf_lines = [
        {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "created_at": "2026-10-19T10:00:00Z", "content": "<USER_REQUEST>Inspect application</USER_REQUEST>"},
        {"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "created_at": "2026-10-19T10:00:01Z", "thinking": "Running ls to list files", "tool_calls": [{"name": "run_command", "args": {"CommandLine": "ls -la"}}]},
        {"step_index": 2, "source": "SYSTEM", "type": "TOOL_RESULT", "created_at": "2026-10-19T10:00:02Z", "content": "total 4\n-rw-r--r-- app.py"},
        {"step_index": 3, "source": "MODEL", "type": "PLANNER_RESPONSE", "created_at": "2026-10-19T10:00:03Z", "thinking": "Inspecting app.py", "tool_calls": [{"name": "view_file", "args": {"AbsolutePath": "/test/app.py"}}]},
        {"step_index": 4, "source": "SYSTEM", "type": "TOOL_RESULT", "created_at": "2026-10-19T10:00:04Z", "content": "print('hello')"},
        {"step_index": 5, "source": "MODEL", "type": "PLANNER_RESPONSE", "created_at": "2026-10-19T10:00:05Z", "thinking": "Verification complete", "content": "I have analyzed app.py and found no issues."},
        {"step_index": 6, "source": "USER_EXPLICIT", "type": "USER_INPUT", "created_at": "2026-10-19T10:00:06Z", "content": "<USER_REQUEST>Now ask me a question to confirm deployment.</USER_REQUEST>"},
        {"step_index": 7, "source": "MODEL", "type": "PLANNER_RESPONSE", "created_at": "2026-10-19T10:00:07Z", "tool_calls": [{"name": "ask_question", "args": {"questions": [{"question": "Deploy to prod?", "options": ["yes", "no"]}]}}]},
        {"step_index": 8, "source": "SYSTEM", "type": "TOOL_RESULT", "created_at": "2026-10-19T10:00:08Z", "content": "yes"},
        {"step_index": 9, "source": "MODEL", "type": "PLANNER_RESPONSE", "created_at": "2026-10-19T10:00:09Z", "content": "Deployment confirmed. Proceeding now."},
    ]

    tf_path = logs_dir / "transcript_full.jsonl"
    tc_path = logs_dir / "transcript.jsonl"
    with open(tf_path, "w", encoding="utf-8") as f:
        for item in tf_lines:
            f.write(json.dumps(item) + "\n")
    with open(tc_path, "w", encoding="utf-8") as f:
        for item in tf_lines:
            f.write(json.dumps(item) + "\n")

    history_file = base_dir / "history.jsonl"
    history_file.write_text(
        json.dumps({"conversation_id": conv_id, "timestamp": 1792404000000, "prompt": "Inspect application"}) + "\n" +
        json.dumps({"conversation_id": conv_id, "timestamp": 1792404006000, "prompt": "Now ask me a question to confirm deployment."}) + "\n",
        encoding="utf-8"
    )

    proto_cache = base_dir / "jetbox_summaries_proto.pb"
    proto_cache.write_bytes(b"dummy-proto")

    presence_dir = base_dir / "presence"
    presence_dir.mkdir()
    presence_lock = presence_dir / f"{conv_id}.lock"
    presence_lock.write_text("locked", encoding="utf-8")

    return {
        "base_dir": base_dir,
        "conv_id": conv_id,
        "db_path": db_path,
        "sum_db": sum_db,
        "tf_path": tf_path,
        "tc_path": tc_path,
        "history_file": history_file,
        "presence_lock": presence_lock,
        "proto_cache": proto_cache,
        "steps_dir": steps_dir
    }


# ==============================================================================
# Group 1: Step Classification & Detection of Model Response
# ==============================================================================

def test_detect_step_info_model_response(mock_model_response_session):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]

    # Step 5 is a pure text model response
    cat5, detail5 = detect_step_info(base_dir, conv_id, 5)
    assert cat5 == "model_response"
    assert detail5 is None

    # Step 9 is also a pure text model response
    cat9, detail9 = detect_step_info(base_dir, conv_id, 9)
    assert cat9 == "model_response"
    assert detail9 is None

    # Step 1 is a tool call
    cat1, detail1 = detect_step_info(base_dir, conv_id, 1)
    assert cat1 == "tool_call_command"
    assert detail1 == "run_command"


def test_detect_step_info_model_response_payload_fallback(tmp_path):
    gemini_root = tmp_path / ".gemini" / "antigravity-cli"
    convs_dir = gemini_root / "conversations"
    convs_dir.mkdir(parents=True)
    conv_id = "test-fallback-conv"
    db_path = convs_dir / f"{conv_id}.db"

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, step_payload BLOB)")
    conn.execute("INSERT INTO steps VALUES (?, ?, ?)", (0, 14, b"user prompt"))
    # MODEL_RESPONSE with pure text payload, no known tool signature
    conn.execute("INSERT INTO steps VALUES (?, ?, ?)", (1, 15, b"Just model thoughts and direct answer"))
    conn.commit()
    conn.close()

    cat, detail = detect_step_info(str(gemini_root), conv_id, 1)
    assert cat == "model_response"
    assert detail is None


# ==============================================================================
# Group 2: Core execute_rewind with arm_running (Model Regeneration)
# ==============================================================================

def test_execute_rewind_arm_running(mock_model_response_session, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]
    sum_db = mock_model_response_session["sum_db"]
    db_path = mock_model_response_session["db_path"]

    execute_rewind(base_dir, conv_id, target_step=5, dry_run=False, arm_running=True)
    out, _ = capsys.readouterr()

    assert "Successfully rewound conversation" in out
    assert "to before Model Response (Step 5)" in out
    assert "Armed conversation_summaries.db (step_count=5, status='CASCADE_RUN_STATUS_RUNNING', not_fully_idle=1)" in out

    # Verify summaries DB state
    s_conn = sqlite3.connect(str(sum_db))
    row = s_conn.execute("SELECT step_count, last_user_input_step_index, status, not_fully_idle, raw_summary FROM conversation_summaries WHERE conversation_id = ?", (conv_id,)).fetchone()
    s_conn.close()

    assert row[0] == 5
    assert row[1] == 0
    assert row[2] == "CASCADE_RUN_STATUS_RUNNING"
    assert row[3] == 1
    assert row[4] is None

    # Verify steps DB
    conn = sqlite3.connect(str(db_path))
    remaining = conn.execute("SELECT idx FROM steps").fetchall()
    conn.close()
    assert [r[0] for r in remaining] == [0, 1, 2, 3, 4]


def test_execute_rewind_dry_run_arm_running(mock_model_response_session, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]
    sum_db = mock_model_response_session["sum_db"]
    db_path = mock_model_response_session["db_path"]

    execute_rewind(base_dir, conv_id, target_step=5, dry_run=True, arm_running=True)
    out, _ = capsys.readouterr()

    assert "[DRY-RUN] Rewind target: delete steps >= 5 (before Model Response)" in out
    assert "[DRY-RUN] Steps remaining: 5" in out
    assert "[DRY-RUN] Steps to be deleted from DB: 5" in out
    assert "[DRY-RUN] Armed conversation_summaries.db (step_count=5, status='CASCADE_RUN_STATUS_RUNNING', not_fully_idle=1)" in out

    # Verify no mutation in dry run
    s_conn = sqlite3.connect(str(sum_db))
    status = s_conn.execute("SELECT status FROM conversation_summaries WHERE conversation_id = ?", (conv_id,)).fetchone()[0]
    s_conn.close()
    assert status == "CASCADE_RUN_STATUS_IDLE"

    conn = sqlite3.connect(str(db_path))
    step_count = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
    conn.close()
    assert step_count == 10


# ==============================================================================
# Group 3: CLI Direct Rewind on Pure Model Response Steps
# ==============================================================================

def test_cli_pure_model_response_rewind_e2e(mock_model_response_session, monkeypatch, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "5"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert "Successfully rewound conversation" in out
    assert "to before Model Response (Step 5)" in out
    assert "Armed conversation_summaries.db" in out

    s_conn = sqlite3.connect(str(mock_model_response_session["sum_db"]))
    row = s_conn.execute("SELECT step_count, status, not_fully_idle FROM conversation_summaries WHERE conversation_id = ?", (conv_id,)).fetchone()
    s_conn.close()
    assert row == (5, "CASCADE_RUN_STATUS_RUNNING", 1)


def test_cli_pure_model_response_rejects_intervention_options(mock_model_response_session, monkeypatch, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "5",
        "--replace-text", "altered response text"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1

    _, err = capsys.readouterr()
    assert "Error: Step 5 is a Model Response." in err
    assert "Intervention options" in err
    assert "cannot be used when rewinding without intervention" in err


# ==============================================================================
# Group 4: CLI --before-response on Tool Call Steps
# ==============================================================================

def test_cli_tool_call_without_before_response_fails(mock_model_response_session, monkeypatch, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]

    # Targeting tool call Step 1 without intervention options and without --before-response
    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "1"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1

    _, err = capsys.readouterr()
    assert "Error: Intervene mode on Step 1 ('run_command') requires intervention content." in err


def test_cli_tool_call_with_before_response_succeeds(mock_model_response_session, monkeypatch, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "1",
        "--before-response"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert "Successfully rewound conversation" in out
    assert "to before Model Response (Step 1)" in out
    assert "Armed conversation_summaries.db (step_count=1, status='CASCADE_RUN_STATUS_RUNNING', not_fully_idle=1)" in out

    # Verify database state: steps 1..9 deleted, only step 0 preserved
    conn = sqlite3.connect(str(mock_model_response_session["db_path"]))
    remaining = conn.execute("SELECT idx, step_type FROM steps").fetchall()
    conn.close()
    assert remaining == [(0, 14)]


def test_cli_before_response_multiple_tool_types(mock_model_response_session, monkeypatch, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]

    # Target Step 3 (view_file) with --before-response
    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "3",
        "--before-response"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert "to before Model Response (Step 3)" in out
    assert "step_count=3" in out

    conn = sqlite3.connect(str(mock_model_response_session["db_path"]))
    remaining = conn.execute("SELECT idx FROM steps").fetchall()
    conn.close()
    assert [r[0] for r in remaining] == [0, 1, 2]


def test_cli_before_response_rejects_intervention_options(mock_model_response_session, monkeypatch, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]

    # Attempting to combine --before-response with --simulate-error
    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "1",
        "--before-response",
        "--simulate-error"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1

    _, err = capsys.readouterr()
    assert "Error: Step 1 is a Model Response." in err
    assert "Intervention options" in err


def test_cli_before_response_dry_run(mock_model_response_session, monkeypatch, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "1",
        "--before-response",
        "--dry-run"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert "[DRY-RUN] Rewind target: delete steps >= 1 (before Model Response)" in out
    assert "[DRY-RUN] Armed conversation_summaries.db (step_count=1, status='CASCADE_RUN_STATUS_RUNNING', not_fully_idle=1)" in out

    # Verify no steps deleted on disk
    conn = sqlite3.connect(str(mock_model_response_session["db_path"]))
    count = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
    conn.close()
    assert count == 10


# ==============================================================================
# Group 5: Detailed Step Inspection via show_step and --show-step
# ==============================================================================

def test_show_step_prompt(mock_model_response_session, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]

    show_step(base_dir, conv_id, target_step=0)
    out, _ = capsys.readouterr()

    assert "Step 0: PROMPT (User Input)" in out
    assert "Category:     prompt" in out
    assert "Source:       USER_EXPLICIT" in out
    assert "Inspect application" in out
    assert "Intervention Hints:" in out
    assert "python Antigravity_Rewind.py 0" in out


def test_show_step_model_response(mock_model_response_session, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]

    show_step(base_dir, conv_id, target_step=5)
    out, _ = capsys.readouterr()

    assert "Step 5: MODEL RESPONSE" in out
    assert "Category:     model_response" in out
    assert "Source:       MODEL" in out
    assert "--- Thinking ---" in out
    assert "Verification complete" in out
    assert "I have analyzed app.py and found no issues." in out
    assert "Intervention Hints:" in out
    assert "python Antigravity_Rewind.py 5" in out


def test_show_step_tool_call_and_paired_result(mock_model_response_session, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]

    # Step 1: Tool Call
    show_step(base_dir, conv_id, target_step=1)
    out1, _ = capsys.readouterr()

    assert "Step 1: TOOL CALL (run_command)" in out1
    assert "Paired Res:   Step 2 (tool_result)" in out1
    assert "CommandLine" in out1
    assert "ls -la" in out1
    assert "--before-response" in out1

    # Step 2: Tool Result
    show_step(base_dir, conv_id, target_step=2)
    out2, _ = capsys.readouterr()

    assert "Step 2: TOOL RESULT" in out2
    assert "Paired Call:  Step 1 (run_command)" in out2
    assert "total 4" in out2
    assert "Artifacts:" in out2
    assert "Step Directory: .system_generated/steps/2/ [output.txt]" in out2


def test_show_step_nonexistent_step(mock_model_response_session, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]

    show_step(base_dir, conv_id, target_step=999)
    _, err = capsys.readouterr()
    assert f"Error: Step 999 was not found in conversation {conv_id}." in err


def test_cli_show_step_flag(mock_model_response_session, monkeypatch, capsys):
    base_dir = str(mock_model_response_session["base_dir"])
    conv_id = mock_model_response_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "--show-step", "3"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert "Step 3: TOOL CALL (view_file)" in out
    assert "Paired Res:   Step 4 (tool_result)" in out
    assert "/test/app.py" in out


def test_print_standout_box_formatting(capsys):
    lines = ["Test Header Error", "Detailed explanation of failure"]
    print_standout_box(lines, border_char="!", file=sys.stderr)
    _, err = capsys.readouterr()

    assert "!" * 80 in err
    assert "Test Header Error" in err
    assert "Detailed explanation of failure" in err
