The repository contains **9 test files** totaling **237 test functions** (approx. 7,700 lines of test code) in the `tests/` directory.

The test suite falls into two distinct design generations:
1. **Foundation Test Suite (Functional / Integration Focus)**: `test_rewind_tool.py`, `test_rewind_interventions.py`, `test_model_response_and_inspection.py`, `test_inquiry_and_interceptor.py`, `test_subagent_interventions.py`, `test_task_interventions.py`, and `test_tui.py`.
2. **Adversarial QA Suite (Strict TESTING.md Focus)**: `test_forks.py` and `test_review_bugs.py`.

---

### 1. Test Suite Overview & Catch Map

#### `test_rewind_tool.py` (16 tests)
- **What it catches**:
  - Failures in path resolution for `antigravity-cli` home directories and output destinations.
  - Multi-store truncation synchronization: mismatches when truncating SQLite `steps`, `gen_metadata`, `executor_metadata`, `parent_references`, and `battle_mode_infos` at step cutoffs.
  - Transcript desynchronization: partial or corrupt truncations in `transcript_full.jsonl`, `transcript.jsonl`, chunk files, and `history.jsonl` timestamp filtering.
  - Cache and lock leaks: failures to invalidate binary protobuf cache (`jetbox_summaries_proto.pb`) and delete `.lock` files.
  - Dry-run violations: mutations occurring when `--dry-run` is requested.
- **Fit with TESTING.md**:
  - *Defeat Tautologies*: **Partial fit**. Uses real SQLite and `tmp_path` filesystems (zero `MagicMock`). However, it imports internal serialization and parsing helper functions (`count_or_truncate_jsonl`, `count_or_truncate_chunks`, `count_or_filter_history`) rather than testing solely via the public entry point or using an inline test oracle.
  - *Break Happy-Path Bias*: **Non-compliant**. Focuses on valid synthetic sequences; lacks `hypothesis` property tests and adversarial payload corpus (e.g., null bytes, malformed UTF-8).
  - *Catch Silent Side Effects*: **Partial fit**. Asserts specific exit conditions, but lacks an autouse network connection blocker (`socket.socket.connect`) and peak RAM ceilings.
  - *Enforce Strong Assertions*: **Good**. Compares row counts and file contents, though some CLI tests check partial output strings.

---

#### `test_rewind_interventions.py` (13 tests)
- **What it catches**:
  - Low-level Protobuf wire-format surgery bugs: byte-length miscalculations in varint headers when replacement strings are larger or smaller than originals.
  - Classification errors across Categories A (file read), B (file write), C (command execution), and D (web fetch/search).
  - Workspace mutation leaks: ensures Category A/B database interventions alter the agent's observation history without corrupting or modifying files on the host disk.
  - Step artifact discrepancies: ensures `.system_generated/steps/<idx>/output.txt` and `content.md` are synchronized or pruned when errors or content replacements are injected.
  - Index drift during kickstart cleanup.
- **Fit with TESTING.md**:
  - *Defeat Tautologies*: **Partial fit**. Uses real SQLite and `tmp_path` (zero `MagicMock`), but imports internal Protobuf helpers (`parse_fields`, `serialize_fields`, `replace_in_bytes`).
  - *Break Happy-Path Bias*: **Non-compliant**. Tests small positive/negative length deltas, but lacks `hypothesis` invariants and adversarial string corpora (null bytes, whitespace boundaries).
  - *Catch Silent Side Effects*: **Partial fit**. Tests CLI validation rejections, but lacks network blocking.
  - *Enforce Strong Assertions*: **Good**. Verifies exact SQLite binary payloads and file existence.

---

#### `test_model_response_and_inspection.py` (16 tests)
- **What it catches**:
  - State arming bugs: verifies that rewinding a model response correctly sets session status to `CASCADE_RUN_STATUS_RUNNING` and `not_fully_idle = 1` in `conversation_summaries.db`, while pure prompt rewinds set it to idle.
  - Premature intervention attempts: prevents users from supplying intervention text when rewinding pure model responses or rewinding tool calls without `--before-response`.
  - Inspector (`--show-step`) regressions: ensures raw tool arguments, chain-of-thought `<THINKING>` blocks, and paired tool result steps are displayed without truncation or KeyError crashes on missing steps.
- **Fit with TESTING.md**:
  - *Defeat Tautologies*: **Partial fit**. Zero mocks, but imports internal helper functions.
  - *Break Happy-Path Bias*: **Partial fit**. Tests nonexistent step lookups, but lacks fuzzing and `hypothesis` coverage.
  - *Catch Silent Side Effects*: **Partial fit**. Validates CLI exit codes and error output, but does not block live network calls via fixture.
  - *Enforce Strong Assertions*: **Good**. Verifies exact summary status values and step indices.

---

#### `test_inquiry_and_interceptor.py` (5 tests)
- **What it catches**:
  - Category E (`ask_question`) Protobuf corruptions: validates that changing `--choice` or `--write-in` rewrites both the human-readable observation and the internal binary `AskQuestionEntry` (field 147) and its selected option IDs.
  - Kickstart proxy token leaks: verifies that `KickstartStripper` removes the wake-up prompt and synthetic server restart notice without stripping subsequent agent responses or tool results in multi-turn dialogues.
- **Fit with TESTING.md**:
  - *Defeat Tautologies*: **Partial fit**. Imports internal Protobuf helpers. Defines lightweight dummy request/flow objects for mitmproxy integration (compliant with avoiding `unittest.mock`).
  - *Break Happy-Path Bias*: **Non-compliant**. Covers single and multi-turn scenarios, but lacks property-based testing and corrupt payload inputs.
  - *Catch Silent Side Effects*: **Partial fit**. Confirms CLI validation rejections, but has no live socket patch.
  - *Enforce Strong Assertions*: **High**. Asserts exact list lengths and decoded Protobuf field values (`assert selected_ids == ["2"]`).

---

#### `test_subagent_interventions.py` (24 tests)
- **What it catches**:
  - Category F (subagent orchestration) corruption: ensures synthetic `--crash`, `--complete`, and `--keep-alive` injections rewrite Protobuf field 114 while updating or leaving child subagent state files intact.
  - Subagent message routing failures: prevents crashes when `.system_generated/messages/<uuid>.json` is missing or when subagent metadata descriptors are unavailable.
  - Subagent cascade truncation: ensures child databases and downstream subagent steps are cleanly rolled back when the parent turn is cut.
  - Isolation leaks: confirms `--out-dir` clones do not leak mutations back to parent or child sessions.
- **Fit with TESTING.md**:
  - *Defeat Tautologies*: **Partial fit**. Zero `MagicMock`, but re-assigns internal helpers from `Antigravity_Rewind`.
  - *Break Happy-Path Bias*: **Partial fit**. Tests corrupted/empty payloads (`test_extract_subagent_msg_info_corrupted_or_empty`) and Unicode characters, but lacks `hypothesis` tests.
  - *Catch Silent Side Effects*: **Partial fit**. Validates exception reporting on invalid arguments, but lacks a network-blocking fixture.
  - *Enforce Strong Assertions*: **Good**. Checks exact table counts and message JSON fields.

---

#### `test_task_interventions.py` (22 tests)
- **What it catches**:
  - Category G (asynchronous task) future log leaks: catches the condition where rewinding to before a task finishes leaves its final stdout/stderr in `.system_generated/tasks/task-*.log`. Halts execution until the user resolves the log state (`--task-output` or `--keep-task-log`).
  - Multi-task race conditions: ensures all unfinished tasks are reported when multiple tasks exist downstream, preventing partial log resolutions.
  - Notification-log discrepancies: warns when a system message reports exit code 0 while the actual task log contains failure markers.
  - Flag conflicts: prevents combining mutually exclusive flags like `--keep-task-log` and `--task-output`.
- **Fit with TESTING.md**:
  - *Defeat Tautologies*: **Partial fit**. Tests real databases and task log files, but imports internal helper functions.
  - *Break Happy-Path Bias*: **Partial fit**. Parametrizes task outputs and checks empty/whitespace specs, but lacks `hypothesis` tests.
  - *Catch Silent Side Effects*: **Good**. Explicitly tests anti-swallowing of conflicting flags and invalid specs, but lacks socket blocking.
  - *Enforce Strong Assertions*: **High**. Asserts exact log contents, table counts, and warning strings.

---

#### `test_tui.py` (9 tests)
- **What it catches**:
  - TUI cursor clamping: prevents curses index errors when navigating beyond the top or bottom of the step list.
  - Step classification rendering: ensures step badges (`[USR]`, `[CMD]`, `[RES]`, `[PLN]`) and summaries match underlying Protobuf records.
  - Action dispatching: verifies keyboard actions (`R`, `B`, `I`, `D`) trigger appropriate headless rewind or intervention execution pipelines without opening curses popups.
- **Fit with TESTING.md**:
  - *Defeat Tautologies*: **Compliant**. Tests only public TUI classes (`SessionDataLoader`, `RewindTUIApp`) with real SQLite data; no internal helper imports or mocks.
  - *Break Happy-Path Bias*: **Non-compliant**. Does not test adversarial terminal sequences or corrupted databases.
  - *Catch Silent Side Effects*: **Partial fit**. No socket blocking.
  - *Enforce Strong Assertions*: **High**. Asserts exact badge strings, line counts, and state flags.

---

#### `test_forks.py` (62 tests)
- **What it catches**:
  - Multiverse isolation leaks: ensures `--fork <branch>` clones all SQLite databases, transcripts, chunk files, task logs, and subagent trees into a new UUID, leaving the original conversation completely unmodified.
  - Recursive subagent tree decoupling: ensures child subagents are cloned and re-pointed to the fork ID so resuming the fork cannot pollute the parent or original subagents.
  - Incomplete failure cleanup: verifies that if any step of a fork fails (e.g. disk failure, bad input, corrupted database), all partially copied files and summary entries are atomically deleted.
  - Collision guards: blocks overwriting an existing conversation ID or branch.
  - TUI fork dialog edge cases: verifies text prompt cancellation, log resolution prompts, and dry-run guarantees.
- **Fit with TESTING.md**:
  - *Defeat Tautologies*: **Fully compliant**. Imports only the public functions (`execute_fork`, `main`, etc.). Implements its own clean Protobuf encoding helpers (`bytes_field`, `int_field`, `system_message_payload`) inside the test file instead of importing implementation helpers. Zero mocks.
  - *Break Happy-Path Bias*: **Fully compliant**. Uses `hypothesis` property-based testing (`@given(st.text())`) to verify round-trip payload serialization and metamorphic stability (`test_replacement_text_round_trips_through_the_fork_payload`). Parametrizes across every step type and category.
  - *Catch Silent Side Effects*: **Fully compliant**. Features an `autouse=True` fixture that monkeypatches `socket.socket.connect` to guarantee zero network traffic. Tests atomic rollback upon engine failure.
  - *Enforce Strong Assertions*: **Fully compliant**. Performs full structural assertions on database records, file sizes, and directory trees.

---

#### `test_review_bugs.py` (70 tests)
- **What it catches**:
  - 100 KB Chunk splitting bugs: catches crashes when multi-byte UTF-8 characters span across the 102,400-byte transcript chunk boundary.
  - Memory exhaustion: catches excessive memory usage when rewriting large transcripts using `tracemalloc`.
  - Malformed and damaged chunks: ensures corrupted chunk bytes are preserved without crashing the rewind engine.
  - Target isolation: ensures search-and-replace interventions mutate only the tool observation field without corrupting adjacent Protobuf metadata, step arguments, or other occurrences of the search string.
  - Formatted view preservation: prevents line-number drift when editing `view_file` results.
  - Subagent database resilience: prevents modifying subagent reports if the subagent database is corrupt or has missing parent links.
  - Real-world kickstart envelope stripping: catches prompt variations wrapped in `<USER_REQUEST>`, ISO timestamps, and server restart notifications.
  - TUI plain-ASCII bounds: prevents terminal lockups by rejecting multi-byte / non-ASCII text in TUI input boxes.
- **Fit with TESTING.md**:
  - *Defeat Tautologies*: **Fully compliant**. Reuses test-defined Protobuf helpers from `test_forks.py`. Zero `MagicMock` or `unittest.mock`.
  - *Break Happy-Path Bias*: **Fully compliant**. Uses `hypothesis` property tests (`@given`), tests malformed/truncated payloads, corrupted UTF-8 byte sequences, null bytes, and damaged SQLite tables.
  - *Catch Silent Side Effects*: **Fully compliant**. Inherits `block_network` (`socket.socket.connect`) from `test_forks.py`. Enforces a strict peak RAM ceiling via `tracemalloc` (`test_rewriting_chunks_uses_memory_in_proportion_to_the_transcript`). Asserts specific exception types.
  - *Enforce Strong Assertions*: **Fully compliant**. Uses exact equality on binary blobs, memory deltas, and file contents.

---

### 2. Compliance Evaluation Against TESTING.md Criteria

| Criteria from `TESTING.md` | Legacy Suite (`test_rewind_tool.py`, `test_rewind_interventions.py`, `test_model_response_and_inspection.py`, `test_inquiry_and_interceptor.py`, `test_subagent_interventions.py`, `test_task_interventions.py`, `test_tui.py`) | Adversarial Suite (`test_forks.py`, `test_review_bugs.py`) |
| :--- | :--- | :--- |
| **1. Defeat Tautologies** | **Partial**: Uses real SQLite/filesystems (zero mocks), but imports internal serialization helpers (`serialize_fields`, `parse_fields`, `replace_in_bytes`). | **Full**: Imports only public APIs; defines independent inline Protobuf encoders inside tests; zero internal mocks. |
| **2. Break Happy-Path Bias** | **Non-compliant**: Focuses on valid synthetic turn sequences; lacks `hypothesis` property-based tests and fuzzing for null bytes or damaged payloads. | **Full**: Uses `hypothesis` (`@given`) for round-trips and invariants; tests split UTF-8 boundaries, corrupt blobs, and truncated payloads. |
| **3. Catch Silent Side Effects** | **Partial**: Tests CLI error branches, but lacks an autouse network connection blocker (`socket.socket.connect`) and RAM ceiling enforcement. | **Full**: Uses `autouse=True` fixture to block live sockets; enforces peak RAM ceilings via `tracemalloc`; tests atomic rollbacks. |
| **4. Enforce Strong Assertions** | **Partial to Good**: Mostly checks database states, but includes occasional substring matches (`assert "..." in output`). | **Full**: Enforces exact structural equality (`==`) on binary payloads, database states, and file trees. |

---

### 3. Summary & Recommendations

1. **`test_forks.py` and `test_review_bugs.py` fully satisfy the strict criteria in `TESTING.md`**: They implement independent test oracles, property-based testing with `hypothesis`, strict network socket blocking, peak RAM ceilings with `tracemalloc`, and zero internal mocks.
2. **The 7 foundation test files deviate from `TESTING.md` in three specific ways**:
   - They import internal Protobuf helper functions (`serialize_fields`, `parse_fields`, `replace_in_bytes`) rather than using test-local oracles.
   - They lack `hypothesis` property-based invariants and adversarial edge-case inputs (null bytes, corrupted chunks).
   - They lack the global `block_network` fixture.

If you would like to refactor or harden the earlier test files to bring them into full alignment with `TESTING.md`, please let me know.