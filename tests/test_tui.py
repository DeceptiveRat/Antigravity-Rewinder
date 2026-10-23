import curses
import json
import os
import sqlite3
import sys
from pathlib import Path
import pytest

import rewinding.Antigravity_Rewind as rewind_mod
from rewinding.tui import StepItem, SessionDataLoader, RewindTUIApp


@pytest.fixture
def mock_tui_session(tmp_path):
    """
    Constructs an isolated session with:
      - Step 0: User Prompt (USER_INPUT = 14)
      - Step 1: Tool Call run_command (MODEL_RESPONSE = 15)
      - Step 2: Tool Result run_command (step_type = 132)
      - Step 3: Pure Model Response (MODEL_RESPONSE = 15)
      - Step 4: User Prompt (USER_INPUT = 14)
      - Step 5: Pure Model Response (MODEL_RESPONSE = 15)
    """
    gemini_root = tmp_path / ".gemini"
    base_dir = gemini_root / "antigravity-cli"
    base_dir.mkdir(parents=True)

    conv_id = "test-tui-conv-1234"

    # 1. Database setup
    convs_dir = base_dir / "conversations"
    convs_dir.mkdir()
    db_path = convs_dir / f"{conv_id}.db"

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, status INTEGER, metadata BLOB, step_payload BLOB)")
    cur.execute("CREATE TABLE gen_metadata (idx INTEGER PRIMARY KEY, data BLOB)")
    cur.execute("CREATE TABLE executor_metadata (idx INTEGER PRIMARY KEY, data BLOB)")

    def make_proto_payload(text: str) -> bytes:
        b = text.encode("utf-8")
        sub = rewind_mod.serialize_fields([(4, 2, b)])
        return rewind_mod.serialize_fields([(14, 2, sub)])

    steps_data = [
        (0, 14, 3, None, b"User prompt: Check system"),
        (1, 15, 3, None, b"run_command uname -a"),
        (2, 132, 3, None, make_proto_payload("Linux test-host 6.8.0")),
        (3, 15, 3, None, b"Host is running Linux kernel 6.8.0."),
        (4, 14, 3, None, b"Now check disk"),
        (5, 15, 3, None, b"All drives healthy."),
    ]

    for idx, stype, status, meta, payload in steps_data:
        cur.execute("INSERT INTO steps VALUES (?, ?, ?, ?, ?)", (idx, stype, status, meta, payload))

    for m_idx in range(3):
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
        (conv_id, "TUI Test Session", "Preview text", 6, 4, "CASCADE_RUN_STATUS_IDLE", 0, None)
    )
    s_conn.commit()
    s_conn.close()

    # 3. Transcripts and directories
    brain_dir = base_dir / "brain" / conv_id
    sys_gen = brain_dir / ".system_generated"
    logs_dir = sys_gen / "logs"
    steps_dir = sys_gen / "steps"
    tasks_dir = sys_gen / "tasks"

    for d in [logs_dir, steps_dir, tasks_dir]:
        d.mkdir(parents=True)

    (steps_dir / "2").mkdir()
    (steps_dir / "2" / "output.txt").write_text("Linux test-host 6.8.0\n", encoding="utf-8")

    tf_lines = [
        {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "created_at": "2026-10-23T10:00:00Z", "content": "<USER_REQUEST>Check system</USER_REQUEST>"},
        {"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "created_at": "2026-10-23T10:00:01Z", "thinking": "Let me run uname", "tool_calls": [{"name": "run_command", "args": {"CommandLine": "uname -a"}}]},
        {"step_index": 2, "source": "SYSTEM", "type": "TOOL_RESULT", "created_at": "2026-10-23T10:00:02Z", "content": "Linux test-host 6.8.0"},
        {"step_index": 3, "source": "MODEL", "type": "PLANNER_RESPONSE", "created_at": "2026-10-23T10:00:03Z", "thinking": "Evaluated kernel", "content": "Host is running Linux kernel 6.8.0."},
        {"step_index": 4, "source": "USER_EXPLICIT", "type": "USER_INPUT", "created_at": "2026-10-23T10:00:04Z", "content": "<USER_REQUEST>Now check disk</USER_REQUEST>"},
        {"step_index": 5, "source": "MODEL", "type": "PLANNER_RESPONSE", "created_at": "2026-10-23T10:00:05Z", "content": "All drives healthy."},
    ]

    tf_path = logs_dir / "transcript_full.jsonl"
    tc_path = logs_dir / "transcript.jsonl"
    with open(tf_path, "w", encoding="utf-8") as f:
        for item in tf_lines:
            f.write(json.dumps(item) + "\n")
    with open(tc_path, "w", encoding="utf-8") as f:
        for item in tf_lines:
            f.write(json.dumps(item) + "\n")

    return {
        "base_dir": base_dir,
        "conv_id": conv_id,
        "db_path": db_path,
        "sum_db": sum_db,
        "steps_dir": steps_dir
    }


def test_session_data_loader_indexing(mock_tui_session):
    base_dir = str(mock_tui_session["base_dir"])
    conv_id = mock_tui_session["conv_id"]

    loader = SessionDataLoader(base_dir, conv_id)
    assert len(loader.steps) == 6
    assert loader.title == "TUI Test Session"
    assert loader.status == "CASCADE_RUN_STATUS_IDLE"

    # Step 0: Prompt
    s0 = loader.steps[0]
    assert s0.idx == 0
    assert s0.category == "prompt"
    assert s0.badge == "[USR]"
    assert s0.display_content == "Check system"

    # Step 1: Tool Call
    s1 = loader.steps[1]
    assert s1.idx == 1
    assert s1.category == "tool_call_command"
    assert s1.badge == "[CMD]"
    assert s1.detail == "run_command"
    assert len(s1.tool_calls) == 1
    assert s1.tool_calls[0]["name"] == "run_command"
    assert s1.paired_step == 2
    assert s1.paired_label == "Paired Result"

    # Step 2: Tool Result
    s2 = loader.steps[2]
    assert s2.idx == 2
    assert s2.category == "tool_result"
    assert s2.badge == "[RES]"
    assert s2.paired_step == 1
    assert s2.paired_label == "Paired Call"
    assert len(s2.artifacts) == 1
    assert "output.txt" in s2.artifacts[0]

    # Step 3: Model Response
    s3 = loader.steps[3]
    assert s3.idx == 3
    assert s3.category == "model_response"
    assert s3.badge == "[THK]"
    assert s3.thinking == "Evaluated kernel"


def test_tui_navigation_and_clamping(mock_tui_session):
    base_dir = str(mock_tui_session["base_dir"])
    conv_id = mock_tui_session["conv_id"]

    app = RewindTUIApp(base_dir, conv_id)
    assert app.cursor == 0

    # Move down
    app.navigate(1)
    assert app.cursor == 1
    assert app.get_selected_step().idx == 1

    # Move down multiple
    app.navigate(3)
    assert app.cursor == 4

    # Clamp at maximum
    app.navigate(100)
    assert app.cursor == 5

    # Clamp at minimum
    app.navigate(-100)
    assert app.cursor == 0

    # Inspector scroll
    assert app.inspector_scroll == 0
    app.scroll_inspector(5)
    assert app.inspector_scroll == 5
    app.scroll_inspector(-2)
    assert app.inspector_scroll == 3
    app.scroll_inspector(-10)
    assert app.inspector_scroll == 0


def test_tui_toggle_dry_run(mock_tui_session):
    base_dir = str(mock_tui_session["base_dir"])
    conv_id = mock_tui_session["conv_id"]

    app = RewindTUIApp(base_dir, conv_id, dry_run=False)
    assert app.dry_run is False

    app.toggle_dry_run()
    assert app.dry_run is True
    assert "Dry-run mode is now ON" in app.status_msg

    app.toggle_dry_run()
    assert app.dry_run is False
    assert "Dry-run mode is now OFF" in app.status_msg


def test_tui_format_inspector_lines(mock_tui_session):
    base_dir = str(mock_tui_session["base_dir"])
    conv_id = mock_tui_session["conv_id"]

    app = RewindTUIApp(base_dir, conv_id)
    step1 = app.loader.steps[1]

    lines = app.format_inspector_lines(step1)
    line_texts = [text for text, _ in lines]

    assert any("TOOL CALL (run_command)" in t for t in line_texts)
    assert any("Category:   tool_call_command" in t for t in line_texts)
    assert any("Paired Result: Step 2" in t for t in line_texts)
    assert any("uname -a" in t for t in line_texts)
    assert any("Let me run uname" in t for t in line_texts)
    assert any("[B] to rewind before this tool decision" in t for t in line_texts)


def test_tui_execute_rewind_headless(mock_tui_session):
    base_dir = str(mock_tui_session["base_dir"])
    conv_id = mock_tui_session["conv_id"]
    db_path = mock_tui_session["db_path"]

    app = RewindTUIApp(base_dir, conv_id, dry_run=False)
    # Cursor at Step 4 (Prompt)
    app.cursor = 4

    ok, msg = app.execute_action_rewind(arm_running=False)
    assert ok is True
    assert "Rewound to step 4 successfully" in msg

    # Verify database state: steps 4 and 5 deleted
    conn = sqlite3.connect(str(db_path))
    steps_remaining = [r[0] for r in conn.execute("SELECT idx FROM steps").fetchall()]
    conn.close()
    assert steps_remaining == [0, 1, 2, 3]


def test_tui_execute_before_response_headless(mock_tui_session):
    base_dir = str(mock_tui_session["base_dir"])
    conv_id = mock_tui_session["conv_id"]
    sum_db = mock_tui_session["sum_db"]

    app = RewindTUIApp(base_dir, conv_id, dry_run=False)
    # Target Step 1 (tool call) with arm_running=True (before response)
    app.cursor = 1

    ok, msg = app.execute_action_rewind(arm_running=True)
    assert ok is True
    assert "Rewound before response successfully" in msg

    # Verify summaries DB is armed for regeneration
    s_conn = sqlite3.connect(str(sum_db))
    row = s_conn.execute("SELECT step_count, status, not_fully_idle FROM conversation_summaries WHERE conversation_id = ?", (conv_id,)).fetchone()
    s_conn.close()
    assert row == (1, "CASCADE_RUN_STATUS_RUNNING", 1)


def test_tui_execute_intervene_headless(mock_tui_session):
    base_dir = str(mock_tui_session["base_dir"])
    conv_id = mock_tui_session["conv_id"]

    app = RewindTUIApp(base_dir, conv_id, dry_run=False)

    # 1. Attempting to intervene on a prompt step fails gracefully
    app.cursor = 0
    ok_p, msg_p = app.execute_action_intervene(replace_old="test", replace_new="new")
    assert ok_p is False
    assert "cannot be intervened upon" in msg_p

    # 2. Intervene on tool call Step 1
    app.cursor = 1
    ok_t, msg_t = app.execute_action_intervene(replace_old="Linux", replace_new="FreeBSD")
    assert ok_t is True
    assert "Intervened on step 1 successfully" in msg_t


def test_tui_keyboard_handler_dispatch(mock_tui_session):
    base_dir = str(mock_tui_session["base_dir"])
    conv_id = mock_tui_session["conv_id"]

    app = RewindTUIApp(base_dir, conv_id)
    assert app.cursor == 0
    assert app.is_running is True

    # Arrow Down
    app.handle_input_char(curses.KEY_DOWN)
    assert app.cursor == 1

    # 'j' (vim down)
    app.handle_input_char(ord('j'))
    assert app.cursor == 2

    # 'k' (vim up)
    app.handle_input_char(ord('k'))
    assert app.cursor == 1

    # 'd' (toggle dry run)
    assert app.dry_run is False
    app.handle_input_char(ord('d'))
    assert app.dry_run is True

    # 'q' (quit)
    app.handle_input_char(ord('q'))
    assert app.is_running is False


def test_cli_tui_help_flag(capsys):
    test_argv = ["Antigravity_Rewind.py", "--help"]
    import sys
    sys_argv_backup = sys.argv
    try:
        sys.argv = test_argv
        with pytest.raises(SystemExit) as exc:
            rewind_mod.main()
        assert exc.value.code == 0
        out, _ = capsys.readouterr()
        assert "--tui" in out
        assert "-i" in out
        assert "--interactive" in out
    finally:
        sys.argv = sys_argv_backup
