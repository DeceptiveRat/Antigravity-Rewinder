# Antigravity Rewind (AGY Rewind)

Forensic Rewind and Counterfactual Causal Intervention Tool for Google Antigravity Sessions.

## Overview

Antigravity Rewind allows developers, forensic analysts, and researchers to perform fine-grained time-travel and counterfactual causal interventions on Google Antigravity agent sessions.

It provides two core operations:
1. Regeneration Rewind: Rolls a session back to before a prompt or model response, so the model can produce a new trajectory from the preserved context.
2. Counterfactual Intervention: Overwrites tool observations, command outputs, or system notifications in historical turns while discarding downstream turns, enabling causal divergence analysis.

Either operation can change the session in place, or create a fork that leaves the original untouched.

## Key Features

- Pure Model Response Rewind: Truncates history to immediately before a model response.
- Tool Decision Rewind (--before-response): Rewinds before an autonomous tool call was issued, to evaluate alternate decision paths.
- Non-Destructive Forks (--fork): Creates a new conversation at any supported step and records its lineage. The original conversation and its subagents are not changed.
- Multi-Category Counterfactual Interventions:
  - Category A (File Reads): Modify observation text in read_file, view_file, and list_dir.
  - Category B (File Writes): Alter write_to_file and replace_file_content database observations without touching disk state.
  - Category C (Command Execution): Modify command output or simulate nonzero exit codes (--simulate-error).
  - Category D (Web Content): Intervene on search_web queries and simulate 404 or altered Markdown in read_url_content. `--replace-web-content` writes the saved page (`content.md`), and also changes it inside the stored result, which is what the agent reads. A result that does not hold the saved page is left as it is, with a note.
  - Category E (Interactive Inquiries): Override selected options (--choice) or write-in responses (--write-in) in ask_question.
  - Category F (Subagent Orchestration): Inject synthetic crashes, completions, or keep-alive states into subagent communication turns.
  - Category G (Background Tasks): Manage background tasks, resolve future log leaks (--task-output, --keep-task-log), and catch notification-log discrepancies.
- Low-Level Protobuf Wire-Format Surgery: Decodes, modifies, and re-encodes nested Protobuf messages in SQLite with varint length recalculation.
- Distributed State Synchronization: Coordinates SQLite databases, JSONL transcripts, chunk logs, shell history, step artifacts, presence locks, and protobuf summary caches.
- Non-Destructive Isolation (--out-dir): Clones state into a separate destination directory before mutating, keeping forensic originals pristine.
- Step Inspection (--show-step / --show): Detailed, untruncated terminal inspector showing turn metadata, raw arguments, internal thinking, paired results, and context-aware CLI examples.

## Installation & Requirements

Requires Python 3.10 or higher. Uses standard library modules only for core operation.

For running tests:
```bash
pip install pytest hypothesis
```

## Basic Usage

Rewinds and interventions change the session in place and cannot be undone, unless you use `--fork` or `--out-dir`. Add `--dry-run` to any command to preview it without writing anything.

### Interactive Terminal Scrubber (TUI)
Launch the keyboard-driven visual timeline scrubber (needs a window of at least 60x12):
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> --tui
# or with short flag
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> -i
```
- Up/Down or j/k: Scrub between historical steps
- PageUp/PageDown: Scroll right-hand inspector view
- B: Rewind before a response or tool call (keeps the prior context, in place)
- R: Rewind to a prompt, response or tool call (deletes it and everything after it, in place)
- I: Intervene on a tool call or task notification (asks for the text to find and its replacement, in place)
- F: Fork at the selected step, leaving the original untouched. On tool calls, task notifications, and subagent reports it asks for text that replaces the whole result
- D: Toggle dry-run mode, where a popup shows what would change
- Q: Quit

If a background task's log holds output from steps being discarded, R, B, I, and F ask whether to clear those logs. Answering no cancels the action. For other log content, use `--task-output` on the command line.

The text box accepts one line of plain ASCII, and typing anything else cancels the action. For multi-line or non-ASCII text, use `--replace-text` or `--replace-file` on the command line.

### Inspect a Specific Step
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> --show-step 3
```

### List or Search Steps
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> --list
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> "text to find"
```

### Rewind Before a User Prompt
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 0
```

### Rewind Before a Model Response (Regeneration)
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 1
```

### Rewind Before a Tool Call Decision
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 3 --before-response
```

### Counterfactual Command Error Simulation
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 1 --simulate-error 1 --error-output "Permission denied"
```

### Counterfactual File Content Replacement
Target the tool call step, not the tool result that follows it. To replace part of the result:
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 3 --replace-old "original string" --replace-new "modified string"
```
To replace the whole result:
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 3 --replace-text "new result"
```
A search looks only in the result's text, not in the step's metadata or the tool call's arguments. If the text to find is not there, or the new text is empty, the command stops with an error and changes nothing.

A step can make several tool calls at once, with one result step per call. An intervention changes the first call's result and keeps the others.

### Fork at a Step
A fork takes the same options as a change in place, except `--out-dir`, because a fork is always created beside the original. The original conversation is not changed:
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 3 --replace-text "new result" --fork my-branch
```
The fork gets a new conversation ID, which replaces the original's ID throughout its data. It also gets its own copy of every subagent its remaining steps refer to, so changing or resuming the fork cannot affect the original subagents. A fork at a subagent's report stops with an error if that subagent's database is missing. If a fork fails, everything it created is removed.

To see forks and their lineage:
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli --tree
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli --list-branches
```

### Background Task Logs
A task log holds the task's final output, even when you rewind to before it finished. The tool has no record of how the log looked at an earlier step, so it stops and asks you to decide what the log should contain:
```bash
# Empty the logs, as if the tasks had printed nothing yet
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 5 --task-output ""
# Provide content for one task
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 5 --task-output task-2="partial output"
# Keep the logs as they are
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 5 --keep-task-log
```
These options change only the logs, not what the agent was told. To change a task's reported status, use `--task-status` or `--task-exit-code`.

### Non-Destructive Rewind to a New Directory
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 1 -o /tmp/experiment_session
```

### Preview Changes Without Writing (Dry-Run)
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 1 --dry-run
```

## Resuming the Rewound Session

Resume with:
```bash
agy --conversation <conversation_id>
```

A resumed session does not continue by itself, even after a rewind that sets its status to running. Antigravity resets the status and waits, so the model only continues once it receives a prompt (tested with agy 1.5.2). That prompt becomes a new step, which is followed by a system message.

To avoid the prompt affecting the result, `src/rewinding/interceptor.py` is a mitmproxy addon that removes a chosen token from outgoing requests. Send a prompt that is exactly the token and nothing else, and clean up afterwards with:
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> --cleanup
```

agy finds its data under `$HOME/.gemini/antigravity-cli`, so `HOME` is what points it at another location:
```bash
HOME="/path/to/other/home" agy --conversation <conversation_id>
```
A copy made with `--out-dir` has to be moved to that layout before it can be resumed, and needs its login files. The tool prints the right command after each change.

## Limitations

- Workspace files are not restored. Only the agent's recorded state changes, so files it wrote stay on disk.
- Only the first call's result can be changed when a step makes several calls.
- Results of subagent management tools such as invoke_subagent cannot be changed.
- Many real results keep no text in the database, for example about two in three `read_file` results. These cannot be changed, and the command stops with an error.
- Step types and Protobuf field numbers are reverse-engineered, so an Antigravity update could change them.

## Optional ACTF Parser

`--list` and text search read the session transcripts. If the ACTF parser (`cli.py`) is placed beside the `rewinding` package, they use its parsed events instead. It is not part of this repository and is not required.

## Running Tests

Run the full pytest test suite from the repository root:
```bash
pytest
```
To skip the tests that drive the TUI in a real pseudo-terminal, which take about 20 seconds:
```bash
pytest -m "not slow"
```
