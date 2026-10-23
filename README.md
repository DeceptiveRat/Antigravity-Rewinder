# Antigravity Rewind (AGY Rewind)

Forensic Rewind and Counterfactual Causal Intervention Tool for Google Antigravity Sessions.

## Overview

Antigravity Rewind allows developers, forensic analysts, and researchers to perform fine-grained time-travel and counterfactual causal interventions on Google Antigravity agent sessions.

It provides two core operations:
1. Regeneration Rewind: Rolls back session state to before a prompt or model response (arming the session so the LLM regenerates a new trajectory from the preserved context).
2. Counterfactual Intervention: Overwrites tool observations, command outputs, or system notifications in historical turns while atomically discarding downstream turns, enabling causal divergence analysis.

## Key Features

- Pure Model Response Rewind: Truncates history to immediately before a model response turn and arms the database so Antigravity resumes by re-evaluating preserved context.
- Tool Decision Rewind (--before-response): Rewinds before an autonomous tool call was issued, allowing evaluation of alternate decision paths without requiring intervention options.
- Multi-Category Counterfactual Interventions:
  - Category A (File Reads): Modify observation text in read_file and view_file.
  - Category B (File Writes): Alter write_to_file and replace_file_content database observations without touching disk state.
  - Category C (Command Execution): Modify bash stdout/stderr or simulate nonzero exit codes (--simulate-error).
  - Category D (Web Content): Intervene on search_web queries and simulate 404 or altered Markdown in read_url_content.
  - Category E (Interactive Inquiries): Override selected options (--choice) or write-in responses (--write-in) in ask_question.
  - Category F (Subagent Orchestration): Inject synthetic crashes, completions, or keep-alive states into subagent communication turns.
  - Category G (Background Tasks): Manage background tasks, resolve future log leaks (--task-output, --keep-task-log), and catch notification-log discrepancies.
- Low-Level Protobuf Wire-Format Surgery: Safely decodes, modifies, and re-encodes nested Protobuf messages in SQLite with dynamic varint length header recalculation.
- Distributed State Synchronization: Simultaneously coordinates SQLite databases, JSONL transcripts, chunk logs, shell history, step artifacts, presence locks, and protobuf summary caches.
- Non-Destructive Isolation (--out-dir): Clones state into a separate destination directory before mutating, keeping forensic originals pristine.
- Step Inspection (--show-step / --show): Detailed, untruncated terminal inspector showing turn metadata, raw arguments, internal thinking, paired results, and context-aware CLI examples.

## Installation & Requirements

Requires Python 3.10 or higher. Uses standard library modules only for core operation.

For running tests:
```bash
pip install pytest
```

## Basic Usage

### 1. Inspect a Specific Step
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> --show-step 3
```

### 2. Rewind Before a User Prompt
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 0
```

### 3. Rewind Before a Model Response (Regeneration)
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 1
```

### 4. Rewind Before a Tool Call Decision
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 3 --before-response
```

### 5. Counterfactual Command Error Simulation
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 1 --simulate-error 1 --error-output "Permission denied"
```

### 6. Counterfactual File Content Replacement
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 3 --replace-old "original string" --replace-new "modified string"
```

### 7. Non-Destructive Rewind to a New Directory
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 1 -o /tmp/experiment_session
```

### 8. Preview Changes Without Writing (Dry-Run)
```bash
python3 Antigravity_Rewind.py -d ~/.gemini/antigravity-cli -c <conversation_id> 1 --dry-run
```

## Resuming the Rewound Session

After executing a rewind, resume Antigravity using:
```bash
JETSKI_APP_DATA_DIR="<path_to_gemini_directory>" agy --conversation <conversation_id>
```

## Running Tests

Run the full pytest test suite from the repository root:
```bash
pytest
```
