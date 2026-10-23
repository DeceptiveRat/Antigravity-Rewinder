import json
import sqlite3
import sys
from pathlib import Path
import pytest

import rewinding.Antigravity_Rewind as rewind_mod
from rewinding.interceptor import KickstartStripper

decode_varint = rewind_mod.decode_varint
encode_varint = rewind_mod.encode_varint
parse_fields = rewind_mod.parse_fields
serialize_fields = rewind_mod.serialize_fields
execute_tool_intervene = rewind_mod.execute_tool_intervene
main = rewind_mod.main


# =========================================================================
# Fixtures: Mock Inquiry (ask_question) Session
# =========================================================================

@pytest.fixture
def mock_inquiry_env(tmp_path):
    """
    Constructs an isolated session with an ask_question step:
      - Step 0: Prompt (USER_INPUT = 14)
      - Step 1: Tool Call ask_question (MODEL_RESPONSE = 15)
      - Step 2: Tool Result ask_question (step_type = 132, Option 1 selected)
    """
    gemini_root = tmp_path / ".gemini"
    base_dir = gemini_root / "antigravity-cli"
    base_dir.mkdir(parents=True)

    conv_id = "test-inquiry-conv-5678"

    # Database setup
    convs_dir = base_dir / "conversations"
    convs_dir.mkdir()
    db_path = convs_dir / f"{conv_id}.db"

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, status INTEGER, "
        "render_info BLOB, step_payload BLOB)"
    )
    cur.execute("CREATE TABLE gen_metadata (idx INTEGER PRIMARY KEY, data BLOB)")
    cur.execute("CREATE TABLE executor_metadata (idx INTEGER PRIMARY KEY, data BLOB)")
    cur.execute("CREATE TABLE parent_references (idx INTEGER PRIMARY KEY, data BLOB)")
    cur.execute("CREATE TABLE battle_mode_infos (idx INTEGER PRIMARY KEY, data BLOB)")

    questions_json = json.dumps({
        "questions": [
            {
                "question": "Which authentication approach would you like to implement?",
                "options": [
                    "JWT with refresh tokens",
                    "Session cookies with Redis",
                    "OAuth2 social login"
                ],
                "is_multi_select": False
            }
        ]
    })

    # Tool Call payload (Step 1)
    # Field 7 (tool_call): name = "ask_question", args = questions_json
    tc_sub = serialize_fields([
        (1, 2, b"call_101"),
        (2, 2, b"ask_question"),
        (3, 2, questions_json.encode("utf-8"))
    ])
    step1_payload = serialize_fields([(7, 2, tc_sub)])

    # Tool Result payload (Step 2)
    # Field 14 -> Field 4 (observation string)
    initial_obs = "A1: JWT with refresh tokens"
    res_sub = serialize_fields([(4, 2, initial_obs.encode("utf-8"))])

    # AskQuestionEntry: Field 1 (question), Field 2 (options), Field 4 (selected_option_ids)
    entry_sub = serialize_fields([
        (1, 2, b"Which authentication approach would you like to implement?"),
        (2, 2, b"JWT with refresh tokens"),
        (2, 2, b"Session cookies with Redis"),
        (2, 2, b"OAuth2 social login"),
        (4, 2, b"1")  # Initially Option 1
    ])

    # In Antigravity, AskQuestionEntry is stored within the step_payload protobuf
    step2_payload = serialize_fields([
        (14, 2, res_sub),
        (147, 2, serialize_fields([(1, 2, entry_sub)]))
    ])

    cur.execute("INSERT INTO steps VALUES (0, 14, 1, NULL, ?)", (b"User prompt",))
    cur.execute("INSERT INTO steps VALUES (1, 15, 1, NULL, ?)", (step1_payload,))
    cur.execute("INSERT INTO steps VALUES (2, 132, 3, NULL, ?)", (step2_payload,))
    cur.execute("INSERT INTO gen_metadata VALUES (0, ?)", (b"meta-0",))
    cur.execute("INSERT INTO executor_metadata VALUES (0, ?)", (b"exec-meta-0",))
    cur.execute("INSERT INTO parent_references VALUES (0, ?)", (b"pref-0",))
    cur.execute("INSERT INTO battle_mode_infos VALUES (0, ?)", (b"battle-0",))

    conn.commit()
    conn.close()

    # Transcript directory and logs
    brain_dir = base_dir / "brain" / conv_id
    sys_gen = brain_dir / ".system_generated" / "logs"
    sys_gen.mkdir(parents=True)

    transcript_full = sys_gen / "transcript_full.jsonl"
    transcript_compact = sys_gen / "transcript.jsonl"

    steps_log = [
        {"step_index": 0, "type": "USER_INPUT", "content": "Add authentication"},
        {
            "step_index": 1,
            "type": "PLANNER_RESPONSE",
            "tool_calls": [{"name": "ask_question", "args": {"questions": json.loads(questions_json)["questions"]}}]
        },
        {"step_index": 2, "type": "TOOL_RESULT", "content": initial_obs}
    ]

    with open(transcript_full, "w", encoding="utf-8") as f:
        for item in steps_log:
            f.write(json.dumps(item) + "\n")

    with open(transcript_compact, "w", encoding="utf-8") as f:
        for item in steps_log:
            f.write(json.dumps(item) + "\n")

    return {
        "base_dir": base_dir,
        "conv_id": conv_id,
        "db_path": db_path,
        "transcript_full": transcript_full,
        "transcript_compact": transcript_compact
    }


# =========================================================================
# Module 1: Category E (ask_question) Rewind Intervention Tests
# =========================================================================

def test_inquiry_choice_intervention(mock_inquiry_env):
    """Verifies that intervening with --choice 1 2 updates payload and AskQuestionEntry."""
    base_dir = mock_inquiry_env["base_dir"]
    conv_id = mock_inquiry_env["conv_id"]

    # Intervene on Step 1 (ask_question), selecting Option 2 ("Session cookies with Redis")
    execute_tool_intervene(
        base_dir=base_dir,
        conv_id=conv_id,
        target_step=1,
        tool_name="ask_question",
        choice="1=2",
        dry_run=False
    )

    # 1. Verify observation and AskQuestionEntry in step_payload
    conn = sqlite3.connect(str(mock_inquiry_env["db_path"]))
    row = conn.execute("SELECT step_payload FROM steps WHERE idx=2").fetchone()
    conn.close()

    step_payload_bytes = row[0]
    assert b"A1: Session cookies with Redis" in step_payload_bytes
    assert b"A1: JWT with refresh tokens" not in step_payload_bytes

    # 2. Verify AskQuestionEntry in step_payload has selected_option_ids updated to '2'
    fields = parse_fields(step_payload_bytes)
    f147 = [v for fn, wt, v in fields if fn == 147][0]
    f1 = parse_fields(f147)[0][2]
    entry_fields = parse_fields(f1)
    selected_ids = [val.decode("utf-8") for fn, wt, val in entry_fields if fn == 4]
    assert selected_ids == ["2"]

    # 3. Verify transcript logs
    with open(mock_inquiry_env["transcript_full"], "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    step2_entry = next(l for l in lines if l["step_index"] == 2)
    assert "Session cookies with Redis" in step2_entry["content"]


def test_inquiry_write_in_intervention(mock_inquiry_env):
    """Verifies that intervening with --write-in 1 'Custom Auth' updates payload and AskQuestionEntry."""
    base_dir = mock_inquiry_env["base_dir"]
    conv_id = mock_inquiry_env["conv_id"]

    execute_tool_intervene(
        base_dir=base_dir,
        conv_id=conv_id,
        target_step=1,
        tool_name="ask_question",
        write_in='1="Custom LDAP Auth"',
        dry_run=False
    )

    conn = sqlite3.connect(str(mock_inquiry_env["db_path"]))
    row = conn.execute("SELECT step_payload FROM steps WHERE idx=2").fetchone()
    conn.close()

    step_payload_bytes = row[0]
    assert b"Custom LDAP Auth" in step_payload_bytes

    # Field 5 in AskQuestionEntry holds the write-in response string
    fields = parse_fields(step_payload_bytes)
    f147 = [v for fn, wt, v in fields if fn == 147][0]
    f1 = parse_fields(f147)[0][2]
    entry_fields = parse_fields(f1)
    write_ins = [val.decode("utf-8") for fn, wt, val in entry_fields if fn == 5]
    assert write_ins == ["Custom LDAP Auth"]


def test_inquiry_cli_validation(mock_inquiry_env, monkeypatch, capsys):
    """Verifies CLI argument validation for inquiry options."""
    base_dir = str(mock_inquiry_env["base_dir"])
    conv_id = mock_inquiry_env["conv_id"]

    def run_cli(*args):
        test_argv = ["Antigravity_Rewind.py", "--dir", base_dir, "--conv", conv_id, "--dry-run"] + list(args)
        monkeypatch.setattr(sys, "argv", test_argv)
        with pytest.raises(SystemExit) as exc_info:
            main()
        out, err = capsys.readouterr()
        return exc_info.value.code, out, err

    # 1. Disallow --choice on prompts
    code, out, err = run_cli("0", "--choice", "1=2")
    assert code == 1
    assert "Intervention options" in err and "Prompt" in err

    # 2. Disallow intervening on ask_question without any options
    code, out, err = run_cli("1")
    assert code == 1
    assert "requires intervention content" in err


# =========================================================================
# Module 2: KickstartStripper Interceptor Multi-Turn Tests
# =========================================================================

class MockRequest:
    def __init__(self, body_dict: dict, method="POST", path="/v1/streamGenerateContent"):
        self.method = method
        self.path = path
        self._body = json.dumps(body_dict)

    def get_text(self):
        return self._body

    def set_text(self, text: str):
        self._body = text


class MockFlow:
    def __init__(self, body_dict: dict):
        self.request = MockRequest(body_dict)


def test_interceptor_single_turn_kickstart():
    """Verifies that on Turn 1, the kickstart prompt and synthetic system message are stripped."""
    stripper = KickstartStripper()
    stripper.default_token = "Stop doing everything and standby"

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "Initial task prompt"}]},
            {"role": "model", "parts": [{"text": "Calling tool"}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "ask_question", "response": {}}}]},
            {"role": "user", "parts": [{"text": "Stop doing everything and standby"}]},
            {"role": "user", "parts": [{"text": "<SYSTEM_MESSAGE>\n[Notice] All your subagents have been stopped"}]}
        ]
    }

    flow = MockFlow(payload)
    stripper.request(flow)

    modified_payload = json.loads(flow.request.get_text())
    contents = modified_payload["contents"]

    # Turns 3 and 4 should be stripped, leaving only turns 0, 1, 2
    assert len(contents) == 3
    assert contents[0]["parts"][0]["text"] == "Initial task prompt"
    assert "functionResponse" in contents[2]["parts"][0]


def test_interceptor_multi_turn_preservation():
    """
    Verifies that on Turns 2+, the kickstart turn and notice are stripped,
    BUT subsequent model turns and tool results are preserved intact.
    """
    stripper = KickstartStripper()
    stripper.default_token = "Stop doing everything and standby"

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "Initial task prompt"}]},
            {"role": "model", "parts": [{"text": "Calling tool"}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "ask_question", "response": {}}}]},
            {"role": "user", "parts": [{"text": "Stop doing everything and standby"}]},
            {"role": "user", "parts": [{"text": "<SYSTEM_MESSAGE>\n[Notice] All your subagents have been stopped"}]},
            # Subsequent execution generated by the agent after kickstart
            {"role": "model", "parts": [{"functionCall": {"name": "run_command", "args": {"CommandLine": "ls -la"}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "run_command", "response": {"output": "total 0"}}}]}
        ]
    }

    flow = MockFlow(payload)
    stripper.request(flow)

    modified_payload = json.loads(flow.request.get_text())
    contents = modified_payload["contents"]

    # Kickstart turn (index 3) and system notice (index 4) removed.
    # Turns 0, 1, 2, and the subsequent turns (formerly 5 and 6) must remain!
    assert len(contents) == 5
    assert contents[0]["parts"][0]["text"] == "Initial task prompt"
    assert "functionResponse" in contents[2]["parts"][0]
    assert contents[3]["role"] == "model"
    assert "functionCall" in contents[3]["parts"][0]
    assert contents[4]["role"] == "user"
    assert "functionResponse" in contents[4]["parts"][0]
