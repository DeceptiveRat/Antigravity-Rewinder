import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
import pytest

import rewinding.Antigravity_Rewind as rewind_mod

find_downstream_task_completions = rewind_mod.find_downstream_task_completions
resolve_task_output_text = rewind_mod.resolve_task_output_text
parse_task_output_spec = rewind_mod.parse_task_output_spec
apply_task_log_content = rewind_mod.apply_task_log_content
detect_step_info = rewind_mod.detect_step_info
check_task_log_discrepancy = rewind_mod.check_task_log_discrepancy
execute_task_intervene = rewind_mod.execute_task_intervene
execute_rewind = rewind_mod.execute_rewind
main = rewind_mod.main


@pytest.fixture
def mock_task_session(tmp_path):
    """
    Constructs an isolated, realistic Antigravity session with a background task:
      - Step 0: User Prompt (USER_INPUT = 14)
      - Step 1: Tool Call run_command (sleep 30 && echo "BUILD SUCCESSFUL") (MODEL_RESPONSE = 15)
      - Step 2: Tool Result run_command (step_type = 132, background task launched as task-2)
      - Step 3: Model response: acknowledged background task (MODEL_RESPONSE = 15)
      - Step 4: User Prompt: "Stop doing everything and standby" (USER_INPUT = 14)
      - Step 5: Tool Call manage_task status task-2 (MODEL_RESPONSE = 15)
      - Step 6: Tool Result manage_task (step_type = 132, status RUNNING)
      - Step 7: Model response: waiting for task completion (MODEL_RESPONSE = 15)
      - Step 8: Platform System Message: task-2 finished with result exit code 0 (step_type = 101)
    """
    gemini_root = tmp_path / ".gemini"
    base_dir = gemini_root / "antigravity-cli"
    base_dir.mkdir(parents=True)

    conv_id = "test-task-conv-9876"
    task_id = "task-2"
    full_task_id = f"{conv_id}/{task_id}"

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

    def make_proto_payload(text: str) -> bytes:
        b = text.encode("utf-8")
        sub = rewind_mod.serialize_fields([(4, 2, b)])
        return rewind_mod.serialize_fields([(14, 2, sub)])

    msg8_uuid = "e7492c19-158a-40a2-911e-2ea6f31e3d64"
    msg8_content = f'Task id "{full_task_id}" finished with result:\n\nexit code 0'
    payload_step8 = rewind_mod.build_system_message_payload(
        conv_id,
        8,
        msg8_uuid,
        msg8_content,
        "2026-10-19T05:10:31Z",
        sender=full_task_id,
        priority="MESSAGE_PRIORITY_HIGH"
    )

    steps_data = [
        (0, 14, 3, b"User prompt: run background build"),
        (1, 15, 3, b"Model planner: run_command sleep 30 && echo BUILD SUCCESSFUL"),
        (2, 132, 3, make_proto_payload(f"Tool is running as a background task with task id: {full_task_id}")),
        (3, 15, 3, b"Model planner: launched background task"),
        (4, 14, 3, b"Stop doing everything and standby"),
        (5, 15, 3, b"Model planner: manage_task status"),
        (6, 132, 3, make_proto_payload(f'Task "{task_id}" status: RUNNING')),
        (7, 15, 3, b"Model planner: standing by"),
        (8, 101, 3, payload_step8),
    ]

    for idx, stype, status, payload in steps_data:
        cur.execute("INSERT INTO steps VALUES (?, ?, ?, ?)", (idx, stype, status, payload))
        cur.execute("INSERT INTO parent_references VALUES (?, ?)", (idx, b"pref"))
        cur.execute("INSERT INTO battle_mode_infos VALUES (?, ?)", (idx, b"binfo"))

    for m_idx in range(4):
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
            status TEXT NOT NULL DEFAULT 'CASCADE_RUN_STATUS_DONE',
            not_fully_idle INTEGER NOT NULL DEFAULT 0,
            raw_summary BLOB
        )
    """)
    s_cur.execute(
        "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (conv_id, "Background Task Test", "Task execution", 9, 4, "CASCADE_RUN_STATUS_DONE", 0, b"raw-sum")
    )
    s_conn.commit()
    s_conn.close()

    # 3. Transcripts and task artifacts
    brain_dir = base_dir / "brain" / conv_id
    sys_gen = brain_dir / ".system_generated"
    logs_dir = sys_gen / "logs"
    chunks_full = logs_dir / "chunks" / "transcript_full"
    chunks_comp = logs_dir / "chunks" / "transcript"
    tasks_dir = sys_gen / "tasks"
    steps_dir = sys_gen / "steps"
    messages_dir = sys_gen / "messages"

    for d in [chunks_full, chunks_comp, tasks_dir, steps_dir, messages_dir]:
        d.mkdir(parents=True)

    # Completed task-2.log with 18 bytes
    task_log_file = tasks_dir / f"{task_id}.log"
    task_log_file.write_text("BUILD SUCCESSFUL\n", encoding="utf-8")

    # Step output artifacts
    (steps_dir / "2").mkdir()
    (steps_dir / "2" / "output.txt").write_text(f"Tool is running as a background task with task id: {full_task_id}\n", encoding="utf-8")
    (steps_dir / "6").mkdir()
    (steps_dir / "6" / "output.txt").write_text(f'Task "{task_id}" status: RUNNING\n', encoding="utf-8")

    # System message JSON artifact for Step 8
    msg8_file = messages_dir / f"{msg8_uuid}.json"
    msg8_data = {
        "id": msg8_uuid,
        "sender": full_task_id,
        "priority": "MESSAGE_PRIORITY_HIGH",
        "content": msg8_content,
        "createdAt": "2026-10-19T05:10:31Z"
    }
    msg8_file.write_text(json.dumps(msg8_data, indent=2), encoding="utf-8")

    # Transcripts
    transcript_records = [
        {"step_index": 0, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": "run background build", "created_at": "2026-10-19T05:10:00Z"},
        {
            "step_index": 1,
            "type": "PLANNER_RESPONSE",
            "source": "MODEL",
            "tool_calls": [{"name": "run_command", "args": {"CommandLine": "sleep 30 && echo BUILD SUCCESSFUL"}}],
            "created_at": "2026-10-19T05:10:01Z"
        },
        {"step_index": 2, "type": "TOOL_RESULT", "source": "MODEL", "content": f"Tool is running as a background task with task id: {full_task_id}", "created_at": "2026-10-19T05:10:01Z"},
        {"step_index": 3, "type": "PLANNER_RESPONSE", "source": "MODEL", "content": "launched background task", "created_at": "2026-10-19T05:10:02Z"},
        {"step_index": 4, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": "Stop doing everything and standby", "created_at": "2026-10-19T05:10:03Z"},
        {
            "step_index": 5,
            "type": "PLANNER_RESPONSE",
            "source": "MODEL",
            "tool_calls": [{"name": "manage_task", "args": {"Action": "status", "TaskId": full_task_id}}],
            "created_at": "2026-10-19T05:10:04Z"
        },
        {"step_index": 6, "type": "TOOL_RESULT", "source": "MODEL", "content": f'Task "{task_id}" status: RUNNING', "created_at": "2026-10-19T05:10:04Z"},
        {"step_index": 7, "type": "PLANNER_RESPONSE", "source": "MODEL", "content": "standing by", "created_at": "2026-10-19T05:10:05Z"},
        {"step_index": 8, "type": "SYSTEM_MESSAGE", "source": "MODEL", "content": msg8_content, "created_at": "2026-10-19T05:10:31Z"},
    ]

    tf_path = logs_dir / "transcript_full.jsonl"
    tc_path = logs_dir / "transcript.jsonl"
    with open(tf_path, "w", encoding="utf-8") as f_full, open(tc_path, "w", encoding="utf-8") as f_comp:
        for rec in transcript_records:
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            f_full.write(line)
            f_comp.write(line)

    chunk_full_file = chunks_full / "chunk_0.jsonl"
    chunk_comp_file = chunks_comp / "chunk_0.jsonl"
    shutil.copyfile(tf_path, chunk_full_file)
    shutil.copyfile(tc_path, chunk_comp_file)

    history_path = base_dir / "history.jsonl"
    history_entries = [
        {"session_id": conv_id, "timestamp": 1792386600000, "text": "run background build"},
        {"session_id": conv_id, "timestamp": 1792386603000, "text": "Stop doing everything and standby"},
    ]
    with open(history_path, "w", encoding="utf-8") as hf:
        for h in history_entries:
            hf.write(json.dumps(h) + "\n")

    return {
        "base_dir": base_dir,
        "conv_id": conv_id,
        "task_id": task_id,
        "full_task_id": full_task_id,
        "task_log_file": task_log_file,
    }


def test_detect_step_info_task_categories(mock_task_session):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    cat5, det5 = detect_step_info(base_dir, conv_id, 5)
    assert cat5 == "tool_call_task"
    assert det5 == "manage_task"

    cat8, det8 = detect_step_info(base_dir, conv_id, 8)
    assert cat8 == "task_notification"
    assert "task-2" in det8


def test_find_downstream_task_completions(mock_task_session):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    # At cutoff=6 (Step 5 tool call paired with Step 6 result):
    # Step 8 notification is downstream, task-2 was launched at Step 2, log file has 17 bytes
    downstream = find_downstream_task_completions(base_dir, conv_id, 6)
    assert len(downstream) == 1
    tid, notif_step, sz = downstream[0]
    assert tid == "task-2"
    assert notif_step == 8
    assert sz > 0

    # At cutoff=4 (Prompt step): Step 8 is downstream
    downstream_prompt = find_downstream_task_completions(base_dir, conv_id, 4)
    assert len(downstream_prompt) == 1
    assert downstream_prompt[0][0] == "task-2"

    # At cutoff=8 (Task completion step itself): no downstream completions
    assert find_downstream_task_completions(base_dir, conv_id, 8) == []

    # If log file is 0 bytes, no future leak exists
    mock_task_session["task_log_file"].write_text("", encoding="utf-8")
    assert find_downstream_task_completions(base_dir, conv_id, 6) == []


def test_resolve_task_output_text(tmp_path):
    assert resolve_task_output_text(None) is None
    assert resolve_task_output_text("") == ""
    assert resolve_task_output_text("stdout log line") == "stdout log line"

    f = tmp_path / "custom_task.log"
    f.write_text("file content from disk", encoding="utf-8")
    assert resolve_task_output_text(str(f)) == "file content from disk"


def test_parse_task_output_spec(tmp_path):
    # None
    d, p = parse_task_output_spec(None)
    assert d is None and p == {}

    # Universal string
    d, p = parse_task_output_spec("")
    assert d == "" and p == {}

    d, p = parse_task_output_spec("universal output")
    assert d == "universal output" and p == {}

    # Single per-task spec
    d, p = parse_task_output_spec("task-2=build failed")
    assert d is None and p == {"task-2": "build failed"}

    # Numeric key without 'task-' prefix
    d, p = parse_task_output_spec("4=server running")
    assert d is None and p == {"task-4": "server running"}

    # Quotes stripping
    d, p = parse_task_output_spec('task-2="double quoted"')
    assert d is None and p == {"task-2": "double quoted"}

    d, p = parse_task_output_spec("task-2='single quoted'")
    assert d is None and p == {"task-2": "single quoted"}

    # File path resolution
    custom_log = tmp_path / "custom_task.log"
    custom_log.write_text("loaded from external file", encoding="utf-8")
    d, p = parse_task_output_spec(f"task-2={str(custom_log)}")
    assert d is None and p == {"task-2": "loaded from external file"}

    # Multiple per-task items in a list
    d, p = parse_task_output_spec(["task-2=out2", "task-4=out4"])
    assert d is None and p == {"task-2": "out2", "task-4": "out4"}

    # Mixed universal fallback with per-task override
    d, p = parse_task_output_spec(["fallback", "task-2=specific"])
    assert d == "fallback" and p == {"task-2": "specific"}


def test_apply_task_log_content(mock_task_session):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    apply_task_log_content(base_dir, conv_id, "task-2", "")
    assert mock_task_session["task_log_file"].read_text(encoding="utf-8") == ""

    apply_task_log_content(base_dir, conv_id, "task-2", "new simulated output")
    assert mock_task_session["task_log_file"].read_text(encoding="utf-8") == "new simulated output"


def test_cli_downstream_cutoff_error_without_task_output(mock_task_session, monkeypatch, capsys):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "5",
        "--replace-old", "RUNNING",
        "--replace-new", "DONE"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    _, err = capsys.readouterr()
    assert "Error: Rewind cuts off downstream completion of background task 'task-2'" in err
    assert "The log file 'tasks/task-2.log' currently contains future completion output" in err
    assert "--task-output" in err
    assert "--keep-task-log" in err


def test_cli_downstream_cutoff_with_task_output_empty(mock_task_session, monkeypatch, capsys):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "5",
        "--replace-old", "RUNNING",
        "--replace-new", "DONE",
        "--task-output", ""
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert "Updated task log: tasks/task-2.log (0 bytes)" in out
    assert mock_task_session["task_log_file"].stat().st_size == 0
    assert mock_task_session["task_log_file"].read_text(encoding="utf-8") == ""


def test_cli_downstream_cutoff_with_task_output_custom_content(mock_task_session, monkeypatch, capsys):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    custom_text = "Compiling module A...\nModule A compiled.\n"
    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "5",
        "--replace-old", "RUNNING",
        "--replace-new", "DONE",
        "--task-output", custom_text
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert f"Updated task log: tasks/task-2.log ({len(custom_text)} bytes)" in out
    assert mock_task_session["task_log_file"].read_text(encoding="utf-8") == custom_text


def test_cli_downstream_cutoff_with_keep_task_log(mock_task_session, monkeypatch, capsys):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "5",
        "--replace-old", "RUNNING",
        "--replace-new", "DONE",
        "--keep-task-log"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert "Successfully executed Intervene Mode" in out
    assert mock_task_session["task_log_file"].read_text(encoding="utf-8") == "BUILD SUCCESSFUL\n"


def test_cli_conflict_task_output_and_keep_task_log(mock_task_session, monkeypatch, capsys):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "5",
        "--task-output", "",
        "--keep-task-log"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    _, err = capsys.readouterr()
    assert "Error: Cannot specify both universal --task-output and --keep-task-log simultaneously" in err


def test_cli_prompt_rewind_with_downstream_task(mock_task_session, monkeypatch, capsys):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    # 1. Missing task-output on prompt rewind must fail
    test_argv_fail = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "4"
    ]
    monkeypatch.setattr(sys, "argv", test_argv_fail)
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

    # 2. Specifying --task-output "" on prompt rewind succeeds and clears task log
    test_argv_ok = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "4",
        "--task-output", ""
    ]
    monkeypatch.setattr(sys, "argv", test_argv_ok)
    main()
    out, _ = capsys.readouterr()
    assert "Successfully rewound conversation" in out
    assert "Updated task log: tasks/task-2.log (0 bytes)" in out
    assert mock_task_session["task_log_file"].stat().st_size == 0


def test_command_simulate_error_prunes_task_log(mock_task_session, monkeypatch, capsys):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "1",
        "--simulate-error", "1"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert "Simulated command exit error: 'exit code 1'" in out
    assert "Cleaned background task artifacts for 'task-2'" in out
    assert not mock_task_session["task_log_file"].exists()


def test_task_notification_step_intervene(mock_task_session, monkeypatch, capsys):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "8",
        "--task-status", "failed",
        "--task-exit-code", "1",
        "--task-output", "Segmentation fault (core dumped)"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert "Task Notification" in out
    assert "Task status: failed" in out
    assert "Task exit code: 1" in out
    assert "Updated task log: tasks/task-2.log (32 bytes)" in out
    assert mock_task_session["task_log_file"].read_text(encoding="utf-8") == "Segmentation fault (core dumped)"


@pytest.fixture
def mock_multi_task_session(tmp_path):
    gemini_root = tmp_path / ".gemini"
    base_dir = gemini_root / "antigravity-cli"
    base_dir.mkdir(parents=True)

    conv_id = "test-multi-task-conv-4444"
    task2_id = "task-2"
    task4_id = "task-4"
    full_task2_id = f"{conv_id}/{task2_id}"
    full_task4_id = f"{conv_id}/{task4_id}"

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

    def make_proto_payload(text: str) -> bytes:
        b = text.encode("utf-8")
        sub = rewind_mod.serialize_fields([(4, 2, b)])
        return rewind_mod.serialize_fields([(14, 2, sub)])

    msg8_uuid = "e7492c19-158a-40a2-911e-2ea6f31e3d64"
    msg8_content = f'Task id "{full_task2_id}" finished with result:\n\nexit code 0'
    payload_step8 = rewind_mod.build_system_message_payload(
        conv_id,
        8,
        msg8_uuid,
        msg8_content,
        "2026-10-19T05:10:31Z",
        sender=full_task2_id,
        priority="MESSAGE_PRIORITY_HIGH"
    )

    msg9_uuid = "a1b2c3d4-5678-90ab-cdef-112233445566"
    msg9_content = f'Task id "{full_task4_id}" finished with result:\n\nexit code 0'
    payload_step9 = rewind_mod.build_system_message_payload(
        conv_id,
        9,
        msg9_uuid,
        msg9_content,
        "2026-10-19T05:12:00Z",
        sender=full_task4_id,
        priority="MESSAGE_PRIORITY_HIGH"
    )

    steps_data = [
        (0, 14, 3, b"User prompt: launch services"),
        (1, 15, 3, b"Model planner: run_command build"),
        (2, 132, 3, make_proto_payload(f"Tool is running as a background task with task id: {full_task2_id}")),
        (3, 15, 3, b"Model planner: run_command server"),
        (4, 132, 3, make_proto_payload(f"Tool is running as a background task with task id: {full_task4_id}")),
        (5, 14, 3, b"User prompt: stop and rewind here"),
        (6, 15, 3, b"Model planner: manage_task status task-2"),
        (7, 132, 3, make_proto_payload(f'Task "{task2_id}" status: RUNNING')),
        (8, 101, 3, payload_step8),
        (9, 101, 3, payload_step9),
    ]

    for idx, stype, status, payload in steps_data:
        cur.execute("INSERT INTO steps VALUES (?, ?, ?, ?)", (idx, stype, status, payload))
        cur.execute("INSERT INTO parent_references VALUES (?, ?)", (idx, b"pref"))
        cur.execute("INSERT INTO battle_mode_infos VALUES (?, ?)", (idx, b"binfo"))

    for m_idx in range(5):
        cur.execute("INSERT INTO gen_metadata VALUES (?, ?)", (m_idx, b"gen-meta"))
        cur.execute("INSERT INTO executor_metadata VALUES (?, ?)", (m_idx, b"exec-meta"))

    conn.commit()
    conn.close()

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
            status TEXT NOT NULL DEFAULT 'CASCADE_RUN_STATUS_DONE',
            not_fully_idle INTEGER NOT NULL DEFAULT 0,
            raw_summary BLOB
        )
    """)
    s_cur.execute(
        "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (conv_id, "Multi Task Test", "Multiple tasks running", 10, 5, "CASCADE_RUN_STATUS_DONE", 0, b"raw-sum")
    )
    s_conn.commit()
    s_conn.close()

    brain_dir = base_dir / "brain" / conv_id
    sys_gen = brain_dir / ".system_generated"
    logs_dir = sys_gen / "logs"
    chunks_full = logs_dir / "chunks" / "transcript_full"
    chunks_comp = logs_dir / "chunks" / "transcript"
    tasks_dir = sys_gen / "tasks"
    steps_dir = sys_gen / "steps"
    messages_dir = sys_gen / "messages"

    for d in [chunks_full, chunks_comp, tasks_dir, steps_dir, messages_dir]:
        d.mkdir(parents=True)

    task2_log_file = tasks_dir / f"{task2_id}.log"
    task2_log_file.write_text("BUILD SUCCESSFUL\n", encoding="utf-8")

    task4_log_file = tasks_dir / f"{task4_id}.log"
    task4_log_file.write_text("HTTP SERVER LISTENING ON PORT 8080\n", encoding="utf-8")

    msg8_file = messages_dir / f"{msg8_uuid}.json"
    msg8_file.write_text(json.dumps({
        "id": msg8_uuid,
        "sender": full_task2_id,
        "priority": "MESSAGE_PRIORITY_HIGH",
        "content": msg8_content,
        "createdAt": "2026-10-19T05:10:31Z"
    }), encoding="utf-8")

    msg9_file = messages_dir / f"{msg9_uuid}.json"
    msg9_file.write_text(json.dumps({
        "id": msg9_uuid,
        "sender": full_task4_id,
        "priority": "MESSAGE_PRIORITY_HIGH",
        "content": msg9_content,
        "createdAt": "2026-10-19T05:12:00Z"
    }), encoding="utf-8")

    transcript_full = logs_dir / "transcript_full.jsonl"
    transcript_comp = logs_dir / "transcript.jsonl"
    with open(transcript_full, "w", encoding="utf-8") as tf, open(transcript_comp, "w", encoding="utf-8") as tc:
        for idx in range(10):
            entry = {
                "step_index": idx,
                "created_at": f"2026-10-19T05:00:0{idx}Z",
                "source": "MODEL" if idx in (1, 3, 6) else ("USER_EXPLICIT" if idx in (0, 5) else "SYSTEM"),
                "type": "USER_INPUT" if idx in (0, 5) else ("PLANNER_RESPONSE" if idx in (1, 3, 6) else "SYSTEM_MESSAGE"),
                "content": msg8_content if idx == 8 else (msg9_content if idx == 9 else f"Step {idx}")
            }
            line = json.dumps(entry) + "\n"
            tf.write(line)
            tc.write(line)

    return {
        "base_dir": base_dir,
        "conv_id": conv_id,
        "task2_id": task2_id,
        "task4_id": task4_id,
        "task2_log_file": task2_log_file,
        "task4_log_file": task4_log_file,
    }


def test_cli_multiple_downstream_cutoff_error_listing(mock_multi_task_session, monkeypatch, capsys):
    base_dir = str(mock_multi_task_session["base_dir"])
    conv_id = mock_multi_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "5"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    _, err = capsys.readouterr()
    assert "Error: Rewind cuts off downstream completion of multiple background tasks:" in err
    assert "- 'task-2' (completed at Step 8" in err
    assert "- 'task-4' (completed at Step 9" in err
    assert '--task-output ""' in err
    assert '--task-output task-2=""' in err
    assert "--keep-task-log" in err


def test_cli_multiple_downstream_partial_resolution_error(mock_multi_task_session, monkeypatch, capsys):
    base_dir = str(mock_multi_task_session["base_dir"])
    conv_id = mock_multi_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "5",
        "--task-output", "task-2="
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    _, err = capsys.readouterr()
    assert "Error: Rewind cuts off downstream completion of background task 'task-4'" in err
    assert "task-4.log" in err


def test_cli_multiple_downstream_per_task_output(mock_multi_task_session, monkeypatch, capsys):
    base_dir = str(mock_multi_task_session["base_dir"])
    conv_id = mock_multi_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "5",
        "--task-output", "task-2=",
        "--task-output", "task-4=partial server output"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert "Successfully rewound conversation" in out
    assert "Updated task log: tasks/task-2.log (0 bytes)" in out
    assert "Updated task log: tasks/task-4.log (21 bytes)" in out
    assert mock_multi_task_session["task2_log_file"].read_text(encoding="utf-8") == ""
    assert mock_multi_task_session["task4_log_file"].read_text(encoding="utf-8") == "partial server output"


def test_cli_multiple_downstream_keep_with_override(mock_multi_task_session, monkeypatch, capsys):
    base_dir = str(mock_multi_task_session["base_dir"])
    conv_id = mock_multi_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "5",
        "--keep-task-log",
        "--task-output", "task-4="
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert "Successfully rewound conversation" in out
    assert "Updated task log: tasks/task-4.log (0 bytes)" in out
    assert mock_multi_task_session["task2_log_file"].read_text(encoding="utf-8") == "BUILD SUCCESSFUL\n"
    assert mock_multi_task_session["task4_log_file"].read_text(encoding="utf-8") == ""


def test_cli_multiple_downstream_universal_task_output(mock_multi_task_session, monkeypatch, capsys):
    base_dir = str(mock_multi_task_session["base_dir"])
    conv_id = mock_multi_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "5",
        "--task-output", ""
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert "Successfully rewound conversation" in out
    assert "Updated task log: tasks/task-2.log (0 bytes)" in out
    assert "Updated task log: tasks/task-4.log (0 bytes)" in out
    assert mock_multi_task_session["task2_log_file"].read_text(encoding="utf-8") == ""
    assert mock_multi_task_session["task4_log_file"].read_text(encoding="utf-8") == ""


def test_cli_multiple_downstream_dry_run(mock_multi_task_session, monkeypatch, capsys):
    base_dir = str(mock_multi_task_session["base_dir"])
    conv_id = mock_multi_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "5",
        "--dry-run",
        "--task-output", "task-2=",
        "--task-output", "task-4=server partial"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, _ = capsys.readouterr()
    assert "[DRY-RUN] Would update task log: tasks/task-2.log (0 bytes)" in out
    assert "[DRY-RUN] Would update task log: tasks/task-4.log (14 bytes)" in out
    # Verify disk files remain untouched
    assert mock_multi_task_session["task2_log_file"].read_text(encoding="utf-8") == "BUILD SUCCESSFUL\n"
    assert mock_multi_task_session["task4_log_file"].read_text(encoding="utf-8") == "HTTP SERVER LISTENING ON PORT 8080\n"


def test_check_task_log_discrepancy_unit(mock_task_session):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    # 1. replace_old matches existing log content
    lines = check_task_log_discrepancy(
        target_step=8,
        task_id="task-2",
        output_text=None,
        keep_task_log=False,
        base_dir=base_dir,
        conv_id=conv_id,
        source_base_dir=None,
        replace_old="SUCCESSFUL",
        replace_new="FAILED",
        replace_text=None,
        replace_file=None,
        task_status=None,
        task_exit_code=None
    )
    assert len(lines) > 0
    assert "still contains 'SUCCESSFUL'" in lines[0]

    # 2. keep_task_log suppresses notice
    lines_kept = check_task_log_discrepancy(
        target_step=8,
        task_id="task-2",
        output_text=None,
        keep_task_log=True,
        base_dir=base_dir,
        conv_id=conv_id,
        source_base_dir=None,
        replace_old="SUCCESSFUL",
        replace_new="FAILED",
        replace_text=None,
        replace_file=None,
        task_status=None,
        task_exit_code=None
    )
    assert lines_kept == []

    # 3. output_text provided suppresses notice
    lines_out = check_task_log_discrepancy(
        target_step=8,
        task_id="task-2",
        output_text="custom log",
        keep_task_log=False,
        base_dir=base_dir,
        conv_id=conv_id,
        source_base_dir=None,
        replace_old="SUCCESSFUL",
        replace_new="FAILED",
        replace_text=None,
        replace_file=None,
        task_status=None,
        task_exit_code=None
    )
    assert lines_out == []

    # 4. Status modified without replace_old in log -> generic discrepancy warning
    lines_stat = check_task_log_discrepancy(
        target_step=8,
        task_id="task-2",
        output_text=None,
        keep_task_log=False,
        base_dir=base_dir,
        conv_id=conv_id,
        source_base_dir=None,
        replace_old=None,
        replace_new=None,
        replace_text=None,
        replace_file=None,
        task_status="failed",
        task_exit_code=1
    )
    assert len(lines_stat) > 0
    assert "was not updated" in lines_stat[0]


def test_cli_task_notification_mismatch_warning_non_halting_e2e(mock_task_session, monkeypatch, capsys):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "8",
        "--replace-old", "SUCCESSFUL",
        "--replace-new", "FAILED"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, err = capsys.readouterr()

    # Verify rewind completes without halting
    assert "Successfully executed Intervene Mode" in out
    assert "Armed conversation_summaries.db" in out
    # Verify prominent stderr warning
    assert "!" * 80 in err
    assert "Step 8 task notification was modified, but 'tasks/task-2.log' still contains 'SUCCESSFUL'." in err


def test_cli_task_notification_mismatch_suppression(mock_task_session, monkeypatch, capsys):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    # 1. With --keep-task-log
    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "8",
        "--replace-old", "SUCCESSFUL",
        "--replace-new", "FAILED",
        "--keep-task-log"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, err = capsys.readouterr()
    assert "Successfully executed Intervene Mode" in out
    assert "Step 8 task notification was modified" not in err

    # 2. With --task-output
    test_argv2 = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "8",
        "--replace-old", "SUCCESSFUL",
        "--replace-new", "FAILED",
        "--task-output", "BUILD FAILED MANUALLY"
    ]
    monkeypatch.setattr(sys, "argv", test_argv2)

    main()
    out2, err2 = capsys.readouterr()
    assert "Successfully executed Intervene Mode" in out2
    assert "Step 8 task notification was modified" not in err2
    assert mock_task_session["task_log_file"].read_text(encoding="utf-8") == "BUILD FAILED MANUALLY"


def test_cli_task_notification_mismatch_dry_run(mock_task_session, monkeypatch, capsys):
    base_dir = str(mock_task_session["base_dir"])
    conv_id = mock_task_session["conv_id"]

    test_argv = [
        "Antigravity_Rewind.py",
        "--dir", base_dir,
        "--conv", conv_id,
        "8",
        "--dry-run",
        "--replace-old", "SUCCESSFUL",
        "--replace-new", "FAILED"
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    main()
    out, err = capsys.readouterr()
    assert "[DRY-RUN]" in out
    assert "Step 8 task notification was modified, but 'tasks/task-2.log' still contains 'SUCCESSFUL'." in err
    assert mock_task_session["task_log_file"].read_text(encoding="utf-8") == "BUILD SUCCESSFUL\n"

