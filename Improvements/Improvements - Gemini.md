Here is a structured list of high-impact ways to improve and expand the rewinder, starting with your idea on file system artifacts.

---

### 1. Workspace File System Rollback (Inverse Journaling)

**The Problem**: Currently, rewinding rolls back the agent's **internal memory** (SQLite database, transcripts, task logs), but files created or modified by the agent on the actual workspace hard drive (via `write_file`, `edit_file`, or bash commands like `rm` or `git checkout`) remain in their post-turn state.

**Ways to implement this**:
- **Pre-Mutation Snapshotting (Inverse Diff Journal)**:
  - Before any tool call executing a write operation (`write_file`, `replace_file_content`, or `run_command`), record the pre-mutation state of target files into a local cache (e.g. `.system_generated/fs_snapshots/<step_idx>/`).
  - Upon rewinding to Step $N$, apply reverse patches or restore files created/modified after Step $N$, and delete any new files created by later steps.
- **Git Shadow Tree / Worktree Checkpoints**:
  - If the agent is working inside a Git repository, create a silent, lightweight git commit or ref (e.g. `refs/agy-checkpoints/<step_idx>`) at each turn boundary.
  - Rewinding to Step $N$ simply checks out the tree hash associated with Step $N$, cleanly restoring the exact workspace directory tree.
- **Copy-on-Write (CoW) / OverlayFS**:
  - For advanced environments, run the workspace over an OverlayFS mount. Rewinding drops upper layers created after the target step.

---

### 2. Conversation Forking / Multiverse Branching (Non-Destructive Time Travel)

**The Problem**: Currently, rewinding in-place deletes downstream turns, and rewinding to an `-o` directory creates a disconnected flat copy.

**Improvement**:
- **Tree-Structured History (DAG of Trajectories)**:
  - Instead of deleting steps from SQLite, create a new conversation ID that branches off Step $N$ (`agy-rewind fork <conv_id>@<step_idx>`).
  - Maintain a lineage index where Conversation B links back to Conversation A at Turn $N$.
  - Enables true A/B testing: Run 5 parallel branches from the same turn with different simulated conditions (e.g. Tool Success vs. Tool 404 vs. Tool Timeout) and observe divergent trajectories.

---

### 3. Interactive Terminal UI (TUI Visual Scrubber)

**The Problem**: Users currently must run `--list`, read step numbers, and type command-line arguments manually.

**Improvement**:
- **Curses/Textual-based Visual Timeline**:
  - A split-screen terminal interface showing the turn timeline on the left and full step inspection (thinking, tool arguments, outputs) on the right.
  - Use arrow keys to scrub back and forth through conversation turns.
  - Press a hotkey (e.g. `R`) to rewind directly, `B` to rewind before model response, or `E` to open an inline editor to modify observation text before triggering the rewind.

---

### 4. Automated Counterfactual Fault Injection (Robustness Matrix)

**The Problem**: Testing agent error recovery currently requires manually specifying `--simulate-error` on one step at a time.

**Improvement**:
- **Automated Failure Permutation**:
  - A command like `agy-rewind probe-faults <conv_id>` that iterates through every tool call in the session, automatically clones the session, simulates an error (e.g. exit code 1, permission denied, HTTP 500), and records whether the agent successfully self-healed or looped into a failure state.
  - Generates a resilience report scoring how well the agent handles unexpected tool failures.

---

### 5. Multi-Trajectory Divergence Comparator (`agy-rewind diff`)

**The Problem**: When you rewind and re-run an agent, there is no easy way to see exactly where and how the new trajectory diverged from the old one.

**Improvement**:
- **Session Diff Engine**:
  - `agy-rewind diff <conv_id_original> <conv_id_forked>`:
    - Compares token usage, tool invocation sequences, and reasoning drift.
    - Highlights the exact turn where the agent diverged (e.g. Turn 3: original chose `run_command`, fork chose `search_web`).

---

### 6. Sandbox & Process Checkpointing (Container / CRIU Integration)

**The Problem**: Long-running background processes or daemon tasks cannot be easily restored to an exact in-memory state; we can only clear or manipulate their log files.

**Improvement**:
- **Container / VM Checkpoint & Restore**:
  - If the agent executes inside a Docker container, use Checkpoint/Restore in Userspace (CRIU) or Docker snapshots (`docker commit`) per turn.
  - Rewinding restores not just disk files and database records, but the active process tree, network sockets, and environment variables.

---

### Suggested Next Step

Among these, **Workspace File System Rollback (via Git shadow snapshots or inverse diffs)** and **Conversation Forking (A/B branching)** offer the highest technical value. Which area would you like to explore first?