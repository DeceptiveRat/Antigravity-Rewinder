import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
import pytest

import rewinding.Antigravity_Rewind as rewind_mod
from rewinding.interceptor import KickstartStripper

decode_varint = rewind_mod.decode_varint
encode_varint = rewind_mod.encode_varint
parse_fields = rewind_mod.parse_fields
serialize_fields = rewind_mod.serialize_fields
replace_in_bytes = rewind_mod.replace_in_bytes
detect_step_info = rewind_mod.detect_step_info
extract_subagent_msg_info = rewind_mod.extract_subagent_msg_info
build_system_message_payload = rewind_mod.build_system_message_payload
execute_subagent_intervene = rewind_mod.execute_subagent_intervene
main = rewind_mod.main


def make_subagent_report_payload(conv_id: str, child_id: str, msg_uuid: str, content: str, iso_ts: str = "2026-10-18T08:10:51Z") -> bytes:
    """Builds a realistic protobuf payload for a subagent report (step_type = 101)."""
    f1_text = f"[Message] timestamp={iso_ts} sender={child_id} priority=MESSAGE_PRIORITY_HIGH content={content}"
    f4_msg = serialize_fields([
        (1, 2, msg_uuid.encode("utf-8")),
        (2, 2, conv_id.encode("utf-8")),
        (3, 2, child_id.encode("utf-8")),
        (4, 2, content.encode("utf-8")),
        (12, 2, serialize_fields([
            (1, 2, serialize_fields([
                (1, 2, child_id.encode("utf-8")),
                (3, 2, serialize_fields([
                    (2, 2, b"send_message"),
                    (3, 2, json.dumps({"Message": content, "Recipient": conv_id}).encode("utf-8")),
                ]))
            ]))
        ]))
    ])
    f114 = serialize_fields([
        (1, 2, f1_text.encode("utf-8")),
        (4, 2, f4_msg),
    ])
    root = [
        (1, 0, 101),
        (4, 0, 3),
        (114, 2, f114),
    ]
    return serialize_fields(root)


@pytest.fixture
def mock_subagent_session(tmp_path):
    """
    Constructs an isolated, multi-turn Antigravity session with subagents:
      - Step 0: User prompt (USER_INPUT = 14)
      - Step 1: Model planner response (MODEL_RESPONSE = 15)
      - Step 2: Tool Call run_command ls -la (step_type = 132)
      - Step 3: Model planner response spawning subagent (MODEL_RESPONSE = 15)
      - Step 4: Tool Result invoke_subagent (step_type = 132)
      - Step 5: Model planner response acknowledging subagent (MODEL_RESPONSE = 15)
      - Step 6: Subagent Report 1 (step_type = 101, containing 'Hallowed Be Thy Name')
      - Step 7: Model response reacting to Update 1 (MODEL_RESPONSE = 15)
      - Step 8: Subagent Report 2 (step_type = 101)
    """
    gemini_root = tmp_path / ".gemini"
    base_dir = gemini_root / "antigravity-cli"
    base_dir.mkdir(parents=True)

    conv_id = "test-subagent-parent-1111"
    child_id = "test-subagent-worker-2222"
    msg1_uuid = "69e2fc3a-e4d1-4fea-aa12-7358d4841135"
    # Choose msg2_uuid such that it sorts alphabetically before msg1_uuid
    msg2_uuid = "03d43a6c-5906-410f-b352-4065dd59b034"

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

    msg1_content = "[Progress Update 1]\nInspected: part 1.txt\nLyrics: Hallowed Be Thy Name"
    msg2_content = "[Progress Update 2]\nInspected: part 2.txt\nLyrics: Running low, yeah!"

    payload_step6 = make_subagent_report_payload(conv_id, child_id, msg1_uuid, msg1_content, "2026-10-18T08:10:51Z")
    payload_step8 = make_subagent_report_payload(conv_id, child_id, msg2_uuid, msg2_content, "2026-10-18T08:11:04Z")

    steps_data = [
        (0, 14, 3, b"User prompt: investigate part files"),
        (1, 15, 3, b"Model planner: listing directory"),
        (2, 132, 3, b"Tool result: total 32 part 1.txt part 2.txt"),
        (3, 15, 3, b"Model planner: invoking subagent"),
        (4, 132, 3, json.dumps({"conversationId": child_id}).encode("utf-8")),
        (5, 15, 3, b"Model planner: subagent spawned, now awaiting updates"),
        (6, 101, 3, payload_step6),
        (7, 15, 3, b"Model planner: interesting lyrics from update 1"),
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

    # 2. Conversation Summaries DB
    sum_db = base_dir / "conversation_summaries.db"
    s_conn = sqlite3.connect(str(sum_db))
    s_cur = s_conn.cursor()
    s_cur.execute("""
        CREATE TABLE conversation_summaries (
            conversation_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            preview TEXT NOT NULL DEFAULT '',
            step_count INTEGER NOT NULL DEFAULT 0,
            last_modified_time datetime,
            workspace_uris TEXT,
            status TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            project_id TEXT NOT NULL DEFAULT '',
            agent_name TEXT NOT NULL DEFAULT '',
            parent_conversation_id TEXT NOT NULL DEFAULT '',
            nesting_depth INTEGER NOT NULL DEFAULT 0,
            battle_id TEXT NOT NULL DEFAULT '',
            winning_conversation_id TEXT NOT NULL DEFAULT '',
            not_fully_idle numeric NOT NULL DEFAULT false,
            killed numeric NOT NULL DEFAULT false,
            last_user_input_time datetime,
            last_user_input_step_index INTEGER NOT NULL DEFAULT -1,
            app_data_dir TEXT NOT NULL DEFAULT '',
            raw_summary BLOB,
            group_id TEXT NOT NULL DEFAULT ''
        )
    """)
    s_cur.execute("""
        INSERT INTO conversation_summaries (conversation_id, step_count, status, not_fully_idle, killed, last_user_input_step_index)
        VALUES (?, 9, 'CASCADE_RUN_STATUS_IDLE', 0, 0, 0)
    """, (conv_id,))
    s_cur.execute("""
        INSERT INTO conversation_summaries (conversation_id, step_count, status, not_fully_idle, killed, last_user_input_step_index)
        VALUES (?, 15, 'CASCADE_RUN_STATUS_IDLE', 0, 0, 0)
    """, (child_id,))
    s_conn.commit()
    s_conn.close()

    # 3. Brain directory: logs, chunks, messages, subagents
    brain_dir = base_dir / "brain" / conv_id
    sys_gen_dir = brain_dir / ".system_generated"
    logs_dir = sys_gen_dir / "logs"
    chunks_full = logs_dir / "chunks" / "transcript_full"
    chunks_comp = logs_dir / "chunks" / "transcript"
    messages_dir = sys_gen_dir / "messages"
    subagents_dir = sys_gen_dir / "subagents"

    for d in [chunks_full, chunks_comp, messages_dir, subagents_dir]:
        d.mkdir(parents=True)

    # Subagent descriptor JSON
    sub_desc_file = subagents_dir / f"{child_id}.json"
    with open(sub_desc_file, "w", encoding="utf-8") as f:
        json.dump({
            "conversationId": child_id,
            "role": "Part Files Researcher",
            "type": "research",
            "state": "SUBAGENT_STATE_IDLE"
        }, f, indent=2)

    # Message artifacts (Note: msg2 sorts before msg1 alphabetically!)
    msg1_file = messages_dir / f"{msg1_uuid}.json"
    with open(msg1_file, "w", encoding="utf-8") as f:
        json.dump({
            "id": msg1_uuid,
            "recipient": conv_id,
            "sender": child_id,
            "priority": "MESSAGE_PRIORITY_HIGH",
            "timestamp": "2026-10-18T08:10:51Z",
            "content": msg1_content,
            "sourceMetadata": {
                "tool": {
                    "conversationId": child_id,
                    "stepIndex": 6,
                    "toolCall": {
                        "name": "send_message",
                        "argumentsJson": json.dumps({"Message": msg1_content, "Recipient": conv_id})
                    }
                }
            }
        }, f, indent=2)

    msg2_file = messages_dir / f"{msg2_uuid}.json"
    with open(msg2_file, "w", encoding="utf-8") as f:
        json.dump({
            "id": msg2_uuid,
            "recipient": conv_id,
            "sender": child_id,
            "priority": "MESSAGE_PRIORITY_HIGH",
            "timestamp": "2026-10-18T08:11:04Z",
            "content": msg2_content,
            "sourceMetadata": {
                "tool": {
                    "conversationId": child_id,
                    "stepIndex": 12,
                    "toolCall": {
                        "name": "send_message",
                        "argumentsJson": json.dumps({"Message": msg2_content, "Recipient": conv_id})
                    }
                }
            }
        }, f, indent=2)

    read_file = messages_dir / "read.json"
    with open(read_file, "w", encoding="utf-8") as f:
        json.dump({msg1_uuid: True, msg2_uuid: True}, f)

    # Transcript logs
    tf_path = logs_dir / "transcript_full.jsonl"
    tc_path = logs_dir / "transcript.jsonl"
    log_lines = [
        {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE", "created_at": "2026-10-18T08:09:00Z", "content": "investigate part files"},
        {"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE", "created_at": "2026-10-18T08:09:05Z", "content": "Listing dir"},
        {"step_index": 2, "source": "MODEL", "type": "GENERIC", "status": "DONE", "created_at": "2026-10-18T08:09:10Z", "content": "total 32"},
        {"step_index": 3, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE", "created_at": "2026-10-18T08:09:15Z", "content": "Spawning subagent"},
        {"step_index": 4, "source": "MODEL", "type": "GENERIC", "status": "DONE", "created_at": "2026-10-18T08:09:20Z", "content": "Subagent created"},
        {"step_index": 5, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE", "created_at": "2026-10-18T08:09:25Z", "content": "Awaiting subagent"},
        {"step_index": 6, "source": "SYSTEM", "type": "SYSTEM_MESSAGE", "status": "DONE", "created_at": "2026-10-18T08:10:51Z", "content": f"<SYSTEM_MESSAGE>\n[Message] timestamp=2026-10-18T08:10:51Z sender={child_id} priority=MESSAGE_PRIORITY_HIGH content={msg1_content}\n</SYSTEM_MESSAGE>"},
        {"step_index": 7, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE", "created_at": "2026-10-18T08:10:55Z", "content": "Analyzing update 1"},
        {"step_index": 8, "source": "SYSTEM", "type": "SYSTEM_MESSAGE", "status": "DONE", "created_at": "2026-10-18T08:11:04Z", "content": f"<SYSTEM_MESSAGE>\n[Message] timestamp=2026-10-18T08:11:04Z sender={child_id} priority=MESSAGE_PRIORITY_HIGH content={msg2_content}\n</SYSTEM_MESSAGE>"},
    ]

    for p in [tf_path, tc_path]:
        with open(p, "w", encoding="utf-8") as f:
            for l in log_lines:
                f.write(json.dumps(l) + "\n")

    chunk_tf = chunks_full / "00000000.jsonl"
    chunk_tc = chunks_comp / "00000000.jsonl"
    for cp in [chunk_tf, chunk_tc]:
        with open(cp, "w", encoding="utf-8") as f:
            for l in log_lines:
                f.write(json.dumps(l) + "\n")

    # history.jsonl
    hist_path = base_dir / "history.jsonl"
    with open(hist_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"conversationId": conv_id, "timestamp": 1792310940000, "display": "investigate part files"}) + "\n")

    return {
        "base_dir": str(base_dir),
        "conv_id": conv_id,
        "child_id": child_id,
        "msg1_uuid": msg1_uuid,
        "msg2_uuid": msg2_uuid,
        "msg1_file": str(msg1_file),
        "msg2_file": str(msg2_file),
        "db_path": str(db_path),
        "sum_db": str(sum_db),
        "sub_desc_file": str(sub_desc_file),
        "tf_path": str(tf_path),
        "tc_path": str(tc_path),
        "messages_dir": str(messages_dir)
    }


# =========================================================================
# Group A: Protobuf Parsing & Payload Extraction Tests
# =========================================================================

def test_extract_subagent_msg_info_success():
    """Verifies that extract_subagent_msg_info extracts UUID, sender, and content accurately."""
    conv_id = "parent-conv-abc"
    child_id = "child-conv-xyz"
    msg_id = "msg-12345678-uuid"
    content = "[Progress Update 1] Found part 1 lyrics: Hallowed Be Thy Name"

    payload = make_subagent_report_payload(conv_id, child_id, msg_id, content)
    extracted_mid, extracted_sender, extracted_content = extract_subagent_msg_info(payload)

    assert extracted_mid == msg_id
    assert extracted_sender == child_id
    assert extracted_content == content


def test_extract_subagent_msg_info_corrupted_or_empty():
    """Verifies graceful handling of empty or non-101 payloads without exceptions."""
    assert extract_subagent_msg_info(b"") == (None, None, None)
    assert extract_subagent_msg_info(b"\x08\x0e\x12\x04test") == (None, None, None)
    assert extract_subagent_msg_info(b"\xff\xff\xff\xff") == (None, None, None)


def test_build_system_message_payload_structure():
    """Verifies that build_system_message_payload generates valid protobuf with status=3."""
    conv_id = "parent-uuid"
    step_idx = 7
    msg_uuid = "synth-uuid-999"
    notice = "Worker process died unexpectedly"
    iso_ts = "2026-10-18T08:50:00Z"

    payload = build_system_message_payload(conv_id, step_idx, msg_uuid, notice, iso_ts)
    assert isinstance(payload, bytes)
    assert len(payload) > 0

    fields = parse_fields(payload)
    field_map = {fn: val for fn, wt, val in fields}

    # Field 1 = 101 (step_type SYSTEM_MESSAGE)
    assert field_map.get(1) == 101
    # Field 4 = 3 (status COMPLETED)
    assert field_map.get(4) == 3
    # Field 114 is present and contains notice
    assert 114 in field_map
    assert notice.encode("utf-8") in payload


# =========================================================================
# Group B: Detection & CLI Validation Tests
# =========================================================================

def test_detect_step_info_subagent_report(mock_subagent_session):
    """Verifies detect_step_info classifies step_type 101 as ('subagent_report', sender_uuid)."""
    env = mock_subagent_session
    conn = sqlite3.connect(env["db_path"])
    cur = conn.cursor()

    cat, detail = detect_step_info(env["base_dir"], env["conv_id"], 6)
    assert cat == "subagent_report"
    assert detail == env["child_id"]

    # Verify regular user input
    p_cat, _ = detect_step_info(env["base_dir"], env["conv_id"], 0)
    assert p_cat == "prompt"
    conn.close()


def test_cli_validation_subagent_exit_mutual_exclusion(mock_subagent_session, monkeypatch):
    """Verifies that incompatible tool options are rejected on subagent report steps."""
    env = mock_subagent_session
    test_args = [
        "Antigravity_Rewind.py",
        "-d", env["base_dir"],
        "-c", env["conv_id"],
        "--simulate-error", "timeout",
        "6"
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code != 0


def test_cli_validation_subagent_exit_rejected_on_tools(mock_subagent_session, monkeypatch):
    """Verifies that --subagent-exit is rejected when targeting a prompt or tool call."""
    env = mock_subagent_session
    test_args = [
        "Antigravity_Rewind.py",
        "-d", env["base_dir"],
        "-c", env["conv_id"],
        "--subagent-exit", "crash",
        "0"  # Target step 0 is prompt
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code != 0


def test_cli_validation_keep_alive_requires_replacement(mock_subagent_session, monkeypatch):
    """Verifies that --subagent-exit keep-alive requires replacement content."""
    env = mock_subagent_session
    test_args = [
        "Antigravity_Rewind.py",
        "-d", env["base_dir"],
        "-c", env["conv_id"],
        "--subagent-exit", "keep-alive",
        "6"
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code != 0


# =========================================================================
# Group C: Subagent Exit Modes E2E Tests
# =========================================================================

def test_subagent_exit_crash_e2e(mock_subagent_session):
    """
    Verifies subagent crash intervention:
      1. Step 6 report text is updated and fatal error trailer is attached.
      2. Synthesized Step 7 is created with status = 3 (DONE).
      3. Messages artifact is created and registered in read.json.
      4. Subagent descriptor is set to SUBAGENT_STATE_KILLED.
      5. Summaries DB marks child as CASCADE_RUN_STATUS_KILLED (killed=1).
      6. Downstream steps (original Step 7 and 8) are deleted.
    """
    env = mock_subagent_session
    execute_subagent_intervene(
        base_dir=env["base_dir"],
        conv_id=env["conv_id"],
        target_step=6,
        replace_old="Hallowed Be Thy Name",
        replace_new="Number of the Beast",
        subagent_exit="crash",
        subagent_error_msg="Worker process segfaulted"
    )

    # 1. Check Step 6 & 7 in DB
    conn = sqlite3.connect(env["db_path"])
    cur = conn.cursor()
    rows = cur.execute("SELECT idx, step_type, status, step_payload FROM steps ORDER BY idx").fetchall()
    conn.close()

    indices = [r[0] for r in rows]
    assert indices == [0, 1, 2, 3, 4, 5, 6, 7]  # Steps 8+ deleted, Step 7 synthesized

    step6_payload = rows[6][3]
    assert b"Number of the Beast" in step6_payload
    assert b"[FATAL ERROR]" in step6_payload
    assert b"Hallowed Be Thy Name" not in step6_payload

    # CRITICAL: Step 7 must have status = 3 (DONE), NOT 0 (CANCELED)
    assert rows[7][1] == 101  # SYSTEM_MESSAGE
    assert rows[7][2] == 3    # status = 3 (DONE)
    assert b"Worker process segfaulted" in rows[7][3]

    # 2. Check subagent descriptor
    with open(env["sub_desc_file"], "r", encoding="utf-8") as f:
        desc = json.load(f)
    assert desc["state"] == "SUBAGENT_STATE_KILLED"

    # 3. Check conversation_summaries.db
    s_conn = sqlite3.connect(env["sum_db"])
    s_cur = s_conn.cursor()
    child_summary = s_cur.execute(
        "SELECT status, killed, not_fully_idle FROM conversation_summaries WHERE conversation_id = ?",
        (env["child_id"],)
    ).fetchone()
    parent_summary = s_cur.execute(
        "SELECT status, not_fully_idle, step_count FROM conversation_summaries WHERE conversation_id = ?",
        (env["conv_id"],)
    ).fetchone()
    s_conn.close()

    assert child_summary == ("CASCADE_RUN_STATUS_KILLED", 1, 0)
    assert parent_summary == ("CASCADE_RUN_STATUS_RUNNING", 1, 8)

    # 4. Check message artifact for Step 6
    with open(env["msg1_file"], "r", encoding="utf-8") as f:
        m1 = json.load(f)
    assert "Number of the Beast" in m1["content"]
    assert "[FATAL ERROR]" in m1["content"]
    aj = json.loads(m1["sourceMetadata"]["tool"]["toolCall"]["argumentsJson"])
    assert "Number of the Beast" in aj["Message"]

    # 5. Check synthetic system message artifact in messages/
    msg_files = [f for f in os.listdir(env["messages_dir"]) if f not in ("read.json", f"{env['msg1_uuid']}.json", f"{env['msg2_uuid']}.json")]
    assert len(msg_files) == 1
    synth_path = os.path.join(env["messages_dir"], msg_files[0])
    with open(synth_path, "r", encoding="utf-8") as f:
        sm = json.load(f)
    assert sm["sender"] == "system"
    assert "Worker process segfaulted" in sm["content"]

    # Check read.json registers the synthetic message
    with open(os.path.join(env["messages_dir"], "read.json"), "r", encoding="utf-8") as f:
        read_dict = json.load(f)
    assert read_dict.get(sm["id"]) is True


def test_subagent_exit_complete_e2e(mock_subagent_session):
    """
    Verifies subagent clean completion intervention:
      1. Step 6 report has [TASK COMPLETE] trailer attached.
      2. No synthetic crash notice step is inserted (step count = 7).
      3. Child conversation summary is set to CASCADE_RUN_STATUS_IDLE (killed=1).
    """
    env = mock_subagent_session
    execute_subagent_intervene(
        base_dir=env["base_dir"],
        conv_id=env["conv_id"],
        target_step=6,
        replace_old="Hallowed Be Thy Name",
        replace_new="Children of the Damned",
        subagent_exit="complete"
    )

    conn = sqlite3.connect(env["db_path"])
    cur = conn.cursor()
    rows = cur.execute("SELECT idx, step_type, status, step_payload FROM steps ORDER BY idx").fetchall()
    conn.close()

    assert [r[0] for r in rows] == [0, 1, 2, 3, 4, 5, 6]  # Exactly 7 steps (no Step 7 synthetic)
    assert b"Children of the Damned" in rows[6][3]
    assert b"[TASK COMPLETE]" in rows[6][3]

    # Summaries DB check
    s_conn = sqlite3.connect(env["sum_db"])
    s_cur = s_conn.cursor()
    child_summary = s_cur.execute(
        "SELECT status, killed, not_fully_idle FROM conversation_summaries WHERE conversation_id = ?",
        (env["child_id"],)
    ).fetchone()
    s_conn.close()

    assert child_summary == ("CASCADE_RUN_STATUS_IDLE", 1, 0)


def test_subagent_exit_keep_alive_e2e(mock_subagent_session):
    """
    Verifies keep-alive subagent intervention:
      1. Report content is substituted cleanly without fatal error or completion trailers.
      2. Subagent descriptor and summaries status remain untouched.
    """
    env = mock_subagent_session
    execute_subagent_intervene(
        base_dir=env["base_dir"],
        conv_id=env["conv_id"],
        target_step=6,
        replace_old="Hallowed Be Thy Name",
        replace_new="The Trooper",
        subagent_exit="keep-alive"
    )

    conn = sqlite3.connect(env["db_path"])
    cur = conn.cursor()
    rows = cur.execute("SELECT idx, step_type, status, step_payload FROM steps ORDER BY idx").fetchall()
    conn.close()

    assert [r[0] for r in rows] == [0, 1, 2, 3, 4, 5, 6]
    assert b"The Trooper" in rows[6][3]
    assert b"[FATAL ERROR]" not in rows[6][3]
    assert b"[TASK COMPLETE]" not in rows[6][3]

    # Subagent descriptor remains IDLE
    with open(env["sub_desc_file"], "r", encoding="utf-8") as f:
        desc = json.load(f)
    assert desc["state"] == "SUBAGENT_STATE_IDLE"


# =========================================================================
# Group D: Multi-Message Artifact Exact Matching Test
# =========================================================================

def test_multi_message_artifact_exact_matching(mock_subagent_session):
    """
    Verifies that when multiple messages exist from the same subagent,
    targeting Step 6 decodes Field 114 to match the exact message JSON file,
    even if other messages sort earlier in alphabetical filename order.
    """
    env = mock_subagent_session

    # Confirm msg2 filename sorts before msg1
    assert os.path.basename(env["msg2_file"]) < os.path.basename(env["msg1_file"])

    execute_subagent_intervene(
        base_dir=env["base_dir"],
        conv_id=env["conv_id"],
        target_step=6,
        replace_old="Hallowed Be Thy Name",
        replace_new="Powerslave",
        subagent_exit="keep-alive"
    )

    # msg1 file must be modified with Powerslave
    with open(env["msg1_file"], "r", encoding="utf-8") as f:
        m1 = json.load(f)
    assert "Powerslave" in m1["content"]

    # msg2 file must NOT be modified by Step 6 rewind
    # (Note: msg2 file is deleted downstream if cutoff timestamp matches, or stays intact if pre-cutoff)
    if os.path.exists(env["msg2_file"]):
        with open(env["msg2_file"], "r", encoding="utf-8") as f:
            m2 = json.load(f)
        assert "Powerslave" not in m2["content"]


# =========================================================================
# Group E: Downstream Truncation Integrity Tests
# =========================================================================

def test_downstream_truncation_cleanup(mock_subagent_session):
    """Verifies that transcript files, chunks, and metadata entries after target_step are truncated."""
    env = mock_subagent_session
    execute_subagent_intervene(
        base_dir=env["base_dir"],
        conv_id=env["conv_id"],
        target_step=6,
        replace_old="Hallowed Be Thy Name",
        replace_new="Aces High",
        subagent_exit="crash"
    )

    # Check transcript_full.jsonl
    with open(env["tf_path"], "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    indices = [l["step_index"] for l in lines]
    assert indices == [0, 1, 2, 3, 4, 5, 6, 7]
    assert "Aces High" in lines[6]["content"]
    assert "terminated unexpectedly" in lines[7]["content"]


# =========================================================================
# Group F: Kickstart Interceptor Multi-Part Turn Tests
# =========================================================================

class MockFlow:
    def __init__(self, request_body: dict):
        self.request = MockRequest(request_body)


class MockRequest:
    def __init__(self, data: dict):
        self.method = "POST"
        self.path = "/v1internal:streamGenerateContent?alt=sse"
        self._body = json.dumps(data)

    def get_text(self) -> str:
        return self._body

    def set_text(self, text: str) -> None:
        self._body = text


def test_interceptor_multipart_preserves_system_message():
    """
    Verifies that when a kickstart token shares a user-role turn with a <SYSTEM_MESSAGE>,
    the interceptor strips only the kickstart token part and retains the system message.
    """
    stripper = KickstartStripper()
    stripper.default_token = "Stop doing everything and standby"

    sys_msg_text = "<SYSTEM_MESSAGE>\nSubagent worker terminated unexpectedly\n</SYSTEM_MESSAGE>"
    kickstart_text = "Stop doing everything and standby"

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "Investigate files"}]},
            {"role": "model", "parts": [{"text": "I have spawned a subagent"}]},
            {
                "role": "user",
                "parts": [
                    {"text": sys_msg_text},
                    {"text": kickstart_text}
                ]
            }
        ]
    }

    flow = MockFlow(payload)
    stripper.request(flow)

    modified_payload = json.loads(flow.request.get_text())
    contents = modified_payload["contents"]

    # Must preserve exactly 3 turns
    assert len(contents) == 3
    # The trailing turn must still have role: "user"
    assert contents[-1]["role"] == "user"
    parts = contents[-1]["parts"]
    # Only 1 part must remain (the system message)
    assert len(parts) == 1
    assert parts[0]["text"] == sys_msg_text
    assert kickstart_text not in parts[0]["text"]


def test_interceptor_single_kickstart_turn_removal():
    """Verifies that a standalone kickstart turn is completely removed when it has no other parts."""
    stripper = KickstartStripper()
    stripper.default_token = "Stop doing everything and standby"

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "Run command"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "run_command"}}]},
            {"role": "user", "parts": [{"functionResponse": {"response": {"output": "success"}}}]},
            {"role": "user", "parts": [{"text": "Stop doing everything and standby"}]}
        ]
    }

    flow = MockFlow(payload)
    stripper.request(flow)

    modified_payload = json.loads(flow.request.get_text())
    contents = modified_payload["contents"]

    # Standalone kickstart turn must be removed, leaving 3 turns
    assert len(contents) == 3
    assert contents[-1]["role"] == "user"
    assert "functionResponse" in contents[-1]["parts"][0]


# =========================================================================
# Group G: Dry-Run Mode Test
# =========================================================================

def test_subagent_intervene_dry_run(mock_subagent_session, capsys):
    """Verifies that dry-run outputs planned changes without altering DB or files."""
    env = mock_subagent_session
    before_hash = os.path.getmtime(env["db_path"])

    execute_subagent_intervene(
        base_dir=env["base_dir"],
        conv_id=env["conv_id"],
        target_step=6,
        replace_old="Hallowed Be Thy Name",
        replace_new="Wasted Years",
        subagent_exit="crash",
        dry_run=True
    )

    captured = capsys.readouterr().out
    assert "[DRY-RUN]" in captured
    assert "Intervene Mode: Subagent Report Step 6" in captured
    assert "Will synthesize platform notice Step 7" in captured

    # DB steps must remain intact (9 steps)
    conn = sqlite3.connect(env["db_path"])
    count = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
    conn.close()
    assert count == 9


# =========================================================================
# Group H: Content Replacement Edge Cases
# =========================================================================

def test_subagent_intervene_replace_file_multiline_unicode(mock_subagent_session, tmp_path):
    """Verifies that full report replacement via --replace-file handles markdown, code fences, and Unicode."""
    env = mock_subagent_session
    rep_file = tmp_path / "replacement.md"
    unicode_content = (
        "### Subagent Findings\n"
        "```json\n"
        "{\n"
        '  "status": "완료",\n'
        '  "metrics": {"pi": 3.14159, "chars": "日本語 and ∑ π 音楽"}\n'
        "}\n"
        "```\n"
        "Detailed explanation with markdown tables and quotes."
    )
    with open(rep_file, "w", encoding="utf-8") as f:
        f.write(unicode_content)

    execute_subagent_intervene(
        base_dir=env["base_dir"],
        conv_id=env["conv_id"],
        target_step=6,
        replace_file=str(rep_file),
        subagent_exit="keep-alive"
    )

    conn = sqlite3.connect(env["db_path"])
    payload = conn.execute("SELECT step_payload FROM steps WHERE idx = 6").fetchone()[0]
    conn.close()

    assert unicode_content.encode("utf-8") in payload

    with open(env["msg1_file"], "r", encoding="utf-8") as f:
        m = json.load(f)
    assert unicode_content in m["content"]


def test_subagent_intervene_nonexistent_string_error(mock_subagent_session):
    """Verifies that an error is raised and state is atomically preserved if replace_old is not in report."""
    env = mock_subagent_session
    with pytest.raises(ValueError, match="not found in Step 6 subagent report"):
        execute_subagent_intervene(
            base_dir=env["base_dir"],
            conv_id=env["conv_id"],
            target_step=6,
            replace_old="This string definitely does not exist in report",
            replace_new="Replacement",
            subagent_exit="crash"
        )

    # Verify no steps were deleted and Step 6 is unmodified
    conn = sqlite3.connect(env["db_path"])
    count = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
    payload = conn.execute("SELECT step_payload FROM steps WHERE idx = 6").fetchone()[0]
    conn.close()
    assert count == 9
    assert b"Hallowed Be Thy Name" in payload


# =========================================================================
# Group I: Artifact Resiliency & Missing File Fallbacks
# =========================================================================

def test_missing_message_json_falls_back_to_proto(mock_subagent_session):
    """Verifies that rewinder falls back gracefully to protobuf payload if message JSON file is missing."""
    env = mock_subagent_session
    if os.path.exists(env["msg1_file"]):
        os.remove(env["msg1_file"])

    execute_subagent_intervene(
        base_dir=env["base_dir"],
        conv_id=env["conv_id"],
        target_step=6,
        replace_old="Hallowed Be Thy Name",
        replace_new="Fear of the Dark",
        subagent_exit="keep-alive"
    )

    conn = sqlite3.connect(env["db_path"])
    payload = conn.execute("SELECT step_payload FROM steps WHERE idx = 6").fetchone()[0]
    conn.close()
    assert b"Fear of the Dark" in payload


def test_missing_subagent_descriptor_resilience(mock_subagent_session):
    """Verifies that rewinder does not crash if the subagent descriptor file is absent."""
    env = mock_subagent_session
    if os.path.exists(env["sub_desc_file"]):
        os.remove(env["sub_desc_file"])

    execute_subagent_intervene(
        base_dir=env["base_dir"],
        conv_id=env["conv_id"],
        target_step=6,
        replace_old="Hallowed Be Thy Name",
        replace_new="Flight of Icarus",
        subagent_exit="crash"
    )

    conn = sqlite3.connect(env["db_path"])
    rows = conn.execute("SELECT idx FROM steps").fetchall()
    conn.close()
    assert [r[0] for r in rows] == [0, 1, 2, 3, 4, 5, 6, 7]


# =========================================================================
# Group J: Multi-Subagent Concurrency & Isolation
# =========================================================================

def test_multi_subagent_isolation_on_crash(mock_subagent_session):
    """Verifies that crashing worker A does not modify or kill worker B."""
    env = mock_subagent_session
    worker_b = "test-subagent-worker-3333"

    sub_b_file = Path(env["base_dir"]) / "brain" / env["conv_id"] / ".system_generated" / "subagents" / f"{worker_b}.json"
    with open(sub_b_file, "w", encoding="utf-8") as f:
        json.dump({"conversationId": worker_b, "role": "Worker B", "state": "SUBAGENT_STATE_IDLE"}, f)

    s_conn = sqlite3.connect(env["sum_db"])
    s_conn.execute("""
        INSERT INTO conversation_summaries (conversation_id, step_count, status, not_fully_idle, killed)
        VALUES (?, 5, 'CASCADE_RUN_STATUS_IDLE', 0, 0)
    """, (worker_b,))
    s_conn.commit()
    s_conn.close()

    execute_subagent_intervene(
        base_dir=env["base_dir"],
        conv_id=env["conv_id"],
        target_step=6,
        replace_old="Hallowed Be Thy Name",
        replace_new="Phantom of the Opera",
        subagent_exit="crash"
    )

    # Worker A killed
    with open(env["sub_desc_file"], "r", encoding="utf-8") as f:
        desc_a = json.load(f)
    assert desc_a["state"] == "SUBAGENT_STATE_KILLED"

    # Worker B remains alive
    with open(sub_b_file, "r", encoding="utf-8") as f:
        desc_b = json.load(f)
    assert desc_b["state"] == "SUBAGENT_STATE_IDLE"

    s_conn = sqlite3.connect(env["sum_db"])
    summary_b = s_conn.execute("SELECT status, killed FROM conversation_summaries WHERE conversation_id = ?", (worker_b,)).fetchone()
    s_conn.close()
    assert summary_b == ("CASCADE_RUN_STATUS_IDLE", 0)


# =========================================================================
# Group K: Boundary & Trajectory Edge Cases
# =========================================================================

def test_target_step_is_last_step_in_db(mock_subagent_session):
    """Verifies that rewinding when target_step is already the last step handles step count without off-by-one errors."""
    env = mock_subagent_session
    conn = sqlite3.connect(env["db_path"])
    conn.execute("DELETE FROM steps WHERE idx > 6")
    conn.commit()
    conn.close()

    execute_subagent_intervene(
        base_dir=env["base_dir"],
        conv_id=env["conv_id"],
        target_step=6,
        replace_old="Hallowed Be Thy Name",
        replace_new="Wrathchild",
        subagent_exit="crash"
    )

    conn = sqlite3.connect(env["db_path"])
    rows = conn.execute("SELECT idx, step_type, status FROM steps ORDER BY idx").fetchall()
    conn.close()
    assert [r[0] for r in rows] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert rows[7][1] == 101
    assert rows[7][2] == 3


def test_custom_error_message_with_special_characters(mock_subagent_session):
    """Verifies that subagent error messages containing quotes, colons, and brackets serialize cleanly."""
    env = mock_subagent_session
    special_err = 'Fatal: [SIGSEGV] in worker::process() -> "null pointer dereference: 0x0"'

    execute_subagent_intervene(
        base_dir=env["base_dir"],
        conv_id=env["conv_id"],
        target_step=6,
        replace_old="Hallowed Be Thy Name",
        replace_new="Murders in the Rue Morgue",
        subagent_exit="crash",
        subagent_error_msg=special_err
    )

    conn = sqlite3.connect(env["db_path"])
    rows = conn.execute("SELECT idx, step_payload FROM steps WHERE idx IN (6, 7)").fetchall()
    conn.close()

    assert special_err.encode("utf-8") in rows[0][1]
    assert special_err.encode("utf-8") in rows[1][1]

    synth_files = [f for f in os.listdir(env["messages_dir"]) if f not in ("read.json", f"{env['msg1_uuid']}.json", f"{env['msg2_uuid']}.json")]
    assert len(synth_files) == 1
    with open(os.path.join(env["messages_dir"], synth_files[0]), "r", encoding="utf-8") as f:
        data = json.load(f)
    assert special_err in data["content"]


# =========================================================================
# Group L: Separate Destination Directory Isolation
# =========================================================================

def test_subagent_intervene_separate_out_dir_isolation(mock_subagent_session, tmp_path, monkeypatch):
    """Verifies that rewinding to a separate out_dir leaves source completely unchanged."""
    env = mock_subagent_session
    out_dir = tmp_path / "separate_output" / ".gemini" / "antigravity-cli"

    test_args = [
        "Antigravity_Rewind.py",
        "-d", env["base_dir"],
        "-o", str(out_dir),
        "-c", env["conv_id"],
        "--replace-old", "Hallowed Be Thy Name",
        "--replace-new", "2 Minutes to Midnight",
        "--subagent-exit", "crash",
        "6"
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    main()

    # Source DB must remain pristine
    src_conn = sqlite3.connect(env["db_path"])
    src_rows = src_conn.execute("SELECT idx, step_payload FROM steps ORDER BY idx").fetchall()
    src_conn.close()
    assert len(src_rows) == 9
    assert b"Hallowed Be Thy Name" in src_rows[6][1]
    assert b"2 Minutes to Midnight" not in src_rows[6][1]

    # Destination DB must be modified
    out_db = out_dir / "conversations" / f"{env['conv_id']}.db"
    assert out_db.exists()
    dest_conn = sqlite3.connect(str(out_db))
    dest_rows = dest_conn.execute("SELECT idx, step_payload FROM steps ORDER BY idx").fetchall()
    dest_conn.close()
    assert len(dest_rows) == 8
    assert b"2 Minutes to Midnight" in dest_rows[6][1]
    assert b"Hallowed Be Thy Name" not in dest_rows[6][1]


# =========================================================================
# Group M: Mitmproxy Interceptor Envelope & Token Edge Cases
# =========================================================================

@pytest.mark.parametrize("wrapper_key", ["generateContentRequest", "clientRequest", "modelRequest", "request"])
def test_interceptor_common_wrapper_envelopes(wrapper_key):
    """Verifies locate_contents() handles Cloud Code and Vertex wrapper envelopes."""
    stripper = KickstartStripper()
    stripper.default_token = "Stop doing everything and standby"

    raw_payload = {
        wrapper_key: {
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
                {"role": "model", "parts": [{"text": "Hi"}]},
                {"role": "user", "parts": [{"text": "Stop doing everything and standby"}]}
            ]
        }
    }
    flow = MockFlow(raw_payload)
    stripper.request(flow)

    modified = json.loads(flow.request.get_text())
    assert len(modified[wrapper_key]["contents"]) == 2
    assert modified[wrapper_key]["contents"][-1]["role"] == "model"


def test_interceptor_custom_token_via_env_var(monkeypatch):
    """Verifies that setting AGY_KICKSTART_TOKEN activates the custom token."""
    monkeypatch.setenv("AGY_KICKSTART_TOKEN", "CUSTOM_STANDBY_TRIGGER")
    stripper = KickstartStripper()

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "Task prompt"}]},
            {"role": "user", "parts": [{"text": "CUSTOM_STANDBY_TRIGGER"}]}
        ]
    }
    flow = MockFlow(payload)
    stripper.request(flow)

    modified = json.loads(flow.request.get_text())
    assert len(modified["contents"]) == 1


def test_interceptor_ignores_non_post_or_non_generate_content():
    """Verifies that non-POST and non-generateContent requests are forwarded untouched."""
    stripper = KickstartStripper()
    payload = {"contents": [{"role": "user", "parts": [{"text": "__AGY_KICKSTART__"}]}]}

    # GET request
    flow_get = MockFlow(payload)
    flow_get.request.method = "GET"
    stripper.request(flow_get)
    assert json.loads(flow_get.request.get_text()) == payload

    # Non generateContent path
    flow_other = MockFlow(payload)
    flow_other.request.path = "/v1internal:loadCodeAssist"
    stripper.request(flow_other)
    assert json.loads(flow_other.request.get_text()) == payload

