"""
Interactive Terminal UI (TUI) Visual Scrubber for AGY_rewind.
Provides keyboard-driven step scrubbing, untruncated turn inspection,
and direct one-key rewinding and counterfactual intervention.
"""

import contextlib
from dataclasses import dataclass, field
import io
import json
import os
import re
import sqlite3
import sys
import textwrap
from typing import Any

try:
    import rewinding.Antigravity_Rewind as rewind_mod
except ModuleNotFoundError:
    import Antigravity_Rewind as rewind_mod

detect_step_info = rewind_mod.detect_step_info
execute_rewind = rewind_mod.execute_rewind
execute_tool_intervene = rewind_mod.execute_tool_intervene
execute_task_intervene = rewind_mod.execute_task_intervene
execute_fork = getattr(rewind_mod, "execute_fork", None)
find_downstream_task_completions = rewind_mod.find_downstream_task_completions
find_future_task_logs = rewind_mod.find_future_task_logs
write_future_task_logs = rewind_mod.write_future_task_logs
clean_preview = rewind_mod.clean_preview
STEP_TYPE_USER_INPUT = rewind_mod.STEP_TYPE_USER_INPUT
STEP_TYPE_MODEL_RESPONSE = rewind_mod.STEP_TYPE_MODEL_RESPONSE

# The fork and intervene keys ask for replacement text on these steps
REPLACEABLE_CATEGORIES = rewind_mod.TOOL_INTERVENTION_CATEGORIES + rewind_mod.TASK_INTERVENTION_CATEGORIES
# A fork has its own copy of a subagent, so unlike the intervene key it can also change a subagent's report
TEXT_BOX_REFUSAL = "Cancelled: the text box only accepts plain ASCII. Use the command line."
FORK_TEXT_CATEGORIES = REPLACEABLE_CATEGORIES + ("subagent_report",)
FORKABLE_CATEGORIES = ("prompt", "model_response") + FORK_TEXT_CATEGORIES


def clear_log_spec(future_tasks: list[tuple[str, int, int]]) -> list[str]:
    """Builds a task output spec that empties exactly these logs and leaves every other log alone."""
    return [f"{task_id}=" for task_id, _, _ in future_tasks]


def task_logs_kept_message(future_tasks: list[tuple[str, int, int]]) -> str:
    return f"Cancelled: {len(future_tasks)} task log(s) would keep future output."


def first_line(exc: Exception) -> str:
    """The status bar shows one line, so a multi-line error keeps only its first."""
    return (str(exc).splitlines() or [""])[0]


@dataclass
class StepItem:
    idx: int
    step_type: int
    category: str
    detail: str | None
    source: str
    created_at: str
    type_name: str
    thinking: str
    content: str
    display_content: str
    tool_calls: list[dict] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    paired_step: int | None = None
    paired_label: str | None = None

    @property
    def badge(self) -> str:
        if self.category == "prompt":
            return "[USR]"
        if self.category == "model_response":
            return "[THK]"
        if self.category == "tool_call_command":
            return "[CMD]"
        if self.category == "tool_call_read":
            return "[RED]"
        if self.category == "tool_call_write":
            return "[WRT]"
        if self.category == "tool_call_web":
            return "[WEB]"
        if self.category == "tool_call_inquiry":
            return "[INQ]"
        if self.category == "tool_call_task":
            return "[TSK]"
        if self.category == "task_notification":
            return "[NOT]"
        if self.category == "subagent_report":
            return "[SUB]"
        if self.category == "tool_result":
            return "[RES]"
        return "[STP]"

    @property
    def summary_title(self) -> str:
        if self.category == "prompt":
            return "PROMPT (User Input)"
        if self.category == "model_response":
            return "MODEL RESPONSE"
        if self.category.startswith("tool_call_"):
            return f"TOOL CALL ({self.detail or 'tool'})"
        if self.category == "tool_result":
            return "TOOL RESULT"
        if self.category == "subagent_report":
            return f"SUBAGENT REPORT ({self.detail or 'subagent'})"
        if self.category == "task_notification":
            return f"TASK NOTIFICATION ({self.detail or 'task'})"
        return f"Step {self.idx} ({self.category})"


class SessionDataLoader:
    """Loads and caches conversation history, steps, and artifacts."""

    def __init__(self, base_dir: str, conv_id: str):
        self.base_dir = base_dir
        self.conv_id = conv_id
        self.steps: list[StepItem] = []
        self.title: str = ""
        self.status: str = "IDLE"
        self.step_count: int = 0
        self.last_user_input: int = -1
        self.reload()

    def reload(self):
        """Refreshes conversation steps from SQLite, transcripts, and disk artifacts."""
        self.steps = []
        conv_db_path = os.path.join(self.base_dir, "conversations", f"{self.conv_id}.db")
        summary_db_path = os.path.join(self.base_dir, "conversation_summaries.db")
        transcript_full = os.path.join(self.base_dir, "brain", self.conv_id, ".system_generated", "logs", "transcript_full.jsonl")
        transcript_compact = os.path.join(self.base_dir, "brain", self.conv_id, ".system_generated", "logs", "transcript.jsonl")
        steps_dir = os.path.join(self.base_dir, "brain", self.conv_id, ".system_generated", "steps")
        tasks_dir = os.path.join(self.base_dir, "brain", self.conv_id, ".system_generated", "tasks")

        # 1. Summary DB
        if os.path.exists(summary_db_path):
            try:
                s_conn = sqlite3.connect(summary_db_path)
                row = s_conn.execute(
                    "SELECT title, status, step_count, last_user_input_step_index FROM conversation_summaries WHERE conversation_id = ?",
                    (self.conv_id,)
                ).fetchone()
                s_conn.close()
                if row:
                    self.title = row[0] or ""
                    self.status = row[1] or "IDLE"
                    self.step_count = row[2] or 0
                    self.last_user_input = row[3] if row[3] is not None else -1
            except Exception:
                pass

        # 2. Transcripts map
        raw_map: dict[int, dict] = {}
        tf_src = transcript_full if os.path.exists(transcript_full) else transcript_compact
        if os.path.exists(tf_src):
            try:
                with open(tf_src, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            d = json.loads(line)
                            idx = d.get("step_index")
                            if idx is not None:
                                raw_map[idx] = d
                        except Exception:
                            pass
            except Exception:
                pass

        # 3. Read steps from DB
        db_steps: list[tuple[int, int]] = []
        if os.path.exists(conv_db_path):
            try:
                c_conn = sqlite3.connect(conv_db_path)
                cur = c_conn.cursor()
                db_steps = cur.execute("SELECT idx, step_type FROM steps ORDER BY idx ASC").fetchall()
                c_conn.close()
            except Exception:
                pass

        if not db_steps and raw_map:
            # Fallback if DB missing
            for s_idx in sorted(raw_map.keys()):
                db_steps.append((s_idx, 15 if raw_map[s_idx].get("source") == "MODEL" else 14))

        for idx, stype in db_steps:
            category = "unknown"
            detail = None
            try:
                category, detail = detect_step_info(self.base_dir, self.conv_id, idx)
            except Exception:
                pass

            t_data = raw_map.get(idx, {})
            source = t_data.get("source", "Unknown")
            created_at = t_data.get("created_at", "")
            type_name = t_data.get("type", f"Type {stype}")
            thinking = t_data.get("thinking", "")
            content = t_data.get("content", "")
            tool_calls = t_data.get("tool_calls", [])

            display_content = content
            if category == "prompt" and content:
                display_content = re.sub(r"<USER_SETTINGS_CHANGE>.*?</USER_SETTINGS_CHANGE>", "", display_content, flags=re.DOTALL)
                display_content = re.sub(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", "", display_content, flags=re.DOTALL)
                display_content = re.sub(r"<\/?USER_REQUEST>", "", display_content).strip()

            # Artifacts
            artifacts = []
            st_dir = os.path.join(steps_dir, str(idx))
            if os.path.isdir(st_dir):
                files = os.listdir(st_dir)
                if files:
                    artifacts.append(f"Step Output: .system_generated/steps/{idx}/ [{', '.join(files)}]")

            task_id = None
            if detail and "task-" in str(detail):
                m_t = re.search(r"task-\d+", str(detail))
                if m_t:
                    task_id = m_t.group(0)
            if not task_id and content:
                m_t = re.search(r"task-\d+", content)
                if m_t:
                    task_id = m_t.group(0)
            if task_id:
                t_log = os.path.join(tasks_dir, f"{task_id}.log")
                if os.path.exists(t_log):
                    sz = os.path.getsize(t_log)
                    artifacts.append(f"Task Log: tasks/{task_id}.log ({sz} bytes)")

            # Paired steps
            paired_step = None
            paired_label = None
            if category == "tool_result" and idx > 0:
                paired_step = idx - 1
                paired_label = "Paired Call"
            elif category.startswith("tool_call_"):
                paired_step = idx + 1
                paired_label = "Paired Result"

            item = StepItem(
                idx=idx,
                step_type=stype,
                category=category,
                detail=detail,
                source=source,
                created_at=created_at,
                type_name=type_name,
                thinking=thinking,
                content=content,
                display_content=display_content,
                tool_calls=tool_calls,
                artifacts=artifacts,
                paired_step=paired_step,
                paired_label=paired_label
            )
            self.steps.append(item)


class RewindTUIApp:
    """Terminal UI application state machine and curses renderer."""

    def __init__(self, base_dir: str, conv_id: str, dry_run: bool = False):
        self.base_dir = base_dir
        self.conv_id = conv_id
        self.loader = SessionDataLoader(base_dir, conv_id)
        self.cursor: int = 0
        self.timeline_offset: int = 0
        self.inspector_scroll: int = 0
        self.dry_run: bool = dry_run
        self.status_msg: str = "Ready. Use Up/Down to select step. [B] Before-Resp, [R] Rewind, [I] Intervene, [Q] Quit."
        self.is_running: bool = True
        self.stdscr = None
        self.last_output: list[str] = []
        self.last_notice: list[str] = []

    # ----------------------------------------------------------------------
    # Headless State & Navigation Helpers (Unit Testable)
    # ----------------------------------------------------------------------

    def navigate(self, delta: int):
        """Moves timeline selection cursor."""
        if not self.loader.steps:
            self.cursor = 0
            return
        self.cursor = max(0, min(len(self.loader.steps) - 1, self.cursor + delta))
        self.inspector_scroll = 0

    def scroll_inspector(self, delta: int):
        """Scrolls right-hand inspector pane."""
        self.inspector_scroll = max(0, self.inspector_scroll + delta)

    def toggle_dry_run(self):
        """Toggles dry run mode."""
        self.dry_run = not self.dry_run
        mode = "ON (simulated)" if self.dry_run else "OFF (real modifications)"
        self.status_msg = f"Dry-run mode is now {mode}."

    def get_selected_step(self) -> StepItem | None:
        """Returns the currently highlighted step item."""
        if 0 <= self.cursor < len(self.loader.steps):
            return self.loader.steps[self.cursor]
        return None

    @contextlib.contextmanager
    def _captured_output(self):
        """The engine functions report with print, which would write over the curses screen."""
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                yield
        finally:
            self.last_output = stdout.getvalue().splitlines()
            # Notices arrive in a box, and its border rows mean nothing in a dialog
            self.last_notice = [line for line in stderr.getvalue().splitlines() if len(set(line.strip())) > 1]

    def _rewind_refusal(self, step: StepItem, arm_running: bool) -> str | None:
        """Explains why the selected step cannot be rewound to, or returns None if it can. The command line's rules."""
        if step.category == "model_response" or step.category.startswith("tool_call_"):
            return None
        if step.category == "prompt":
            return f"Cannot rewind before a response at step {step.idx}: it is a prompt. Use R to remove it." if arm_running else None
        if step.category == "tool_result":
            return f"Cannot rewind to a tool result; select the tool call at step {step.idx - 1}."
        return f"Cannot rewind to step {step.idx}: {step.category} steps cannot be rewound to."

    def execute_action_rewind(self, arm_running: bool = False, clear_task_logs: bool = False) -> tuple[bool, str]:
        """Performs rewind action on the currently selected step."""
        step = self.get_selected_step()
        if not step:
            return False, "No step selected."

        refusal = self._rewind_refusal(step, arm_running)
        if refusal:
            self.status_msg = refusal
            return False, refusal

        try:
            future_tasks = find_downstream_task_completions(self.base_dir, self.conv_id, step.idx)
            if future_tasks and not clear_task_logs:
                msg = task_logs_kept_message(future_tasks)
                self.status_msg = msg
                return False, msg

            with self._captured_output():
                execute_rewind(
                    self.base_dir,
                    self.conv_id,
                    target_step=step.idx,
                    dry_run=self.dry_run,
                    arm_running=arm_running
                )
                write_future_task_logs(self.base_dir, self.conv_id, future_tasks, clear_log_spec(future_tasks), dry_run=self.dry_run)
            if not self.dry_run:
                self.loader.reload()
                self.cursor = max(0, min(self.cursor, len(self.loader.steps) - 1))
            action_name = "Rewound before response" if arm_running else f"Rewound to step {step.idx}"
            mode_tag = " [DRY-RUN]" if self.dry_run else ""
            msg = f"{action_name} successfully{mode_tag}."
            self.status_msg = msg
            return True, msg
        except Exception as exc:
            msg = f"Rewind failed: {first_line(exc)}"
            self.status_msg = msg
            return False, msg

    def _intervene_refusal(self, step: StepItem) -> str | None:
        """Explains why the selected step cannot be changed with this key, or returns None if it can."""
        if step.category in REPLACEABLE_CATEGORIES:
            return None
        if step.category == "subagent_report":
            return "Cannot change a subagent report here; use --subagent-exit on the command line."
        if step.category == "tool_call_other":
            return f"Cannot intervene on step {step.idx}: results of {step.detail} cannot be changed."
        return f"Step {step.idx} ({step.category}) cannot be intervened upon."

    def _intervene_on_task(self, step: StepItem, replace_old: str, replace_new: str, replace_text: str,
                           future_tasks: list[tuple[str, int, int]]) -> str | None:
        """Changes a task step through the task code. A blank search text replaces the whole text, as on other tools."""
        whole_text = replace_text or (replace_new if not replace_old else None)
        if not whole_text and not replace_new:
            result_step = step.idx + 1 if step.category == "tool_call_task" else step.idx
            raise ValueError(f"The replacement text is empty, so step {result_step} would not change. Nothing was changed.")
        return execute_task_intervene(
            self.base_dir,
            self.conv_id,
            step.idx,
            category=step.category,
            detail=step.detail,
            replace_old=(replace_old or None),
            replace_new=(replace_new if replace_old else None),
            replace_text=whole_text,
            task_output=(clear_log_spec(future_tasks) if future_tasks else None),
            dry_run=self.dry_run
        )

    def execute_action_intervene(self, replace_old: str = None, replace_new: str = None,
                                  replace_text: str = None, simulate_error: str = None,
                                  clear_task_logs: bool = False) -> tuple[bool, str]:
        """Performs intervention action on the currently selected step."""
        step = self.get_selected_step()
        if not step:
            return False, "No step selected."

        refusal = self._intervene_refusal(step)
        if refusal:
            self.status_msg = refusal
            return False, refusal

        try:
            future_tasks = find_future_task_logs(self.base_dir, self.conv_id, step.idx, step.category, simulate_error)
            if future_tasks and not clear_task_logs:
                msg = task_logs_kept_message(future_tasks)
                self.status_msg = msg
                return False, msg

            target_task = None
            with self._captured_output():
                if step.category in rewind_mod.TASK_INTERVENTION_CATEGORIES:
                    target_task = self._intervene_on_task(step, replace_old, replace_new, replace_text, future_tasks)
                else:
                    execute_tool_intervene(
                        self.base_dir,
                        self.conv_id,
                        target_step=step.idx,
                        tool_name=step.detail or "tool",
                        replace_old=replace_old,
                        replace_new=replace_new,
                        replace_text=replace_text,
                        simulate_error=simulate_error,
                        dry_run=self.dry_run
                    )
                write_future_task_logs(self.base_dir, self.conv_id, future_tasks, clear_log_spec(future_tasks),
                                       skip_task=target_task, dry_run=self.dry_run)
            if not self.dry_run:
                self.loader.reload()
                self.cursor = max(0, min(self.cursor, len(self.loader.steps) - 1))
            mode_tag = " [DRY-RUN]" if self.dry_run else ""
            msg = f"Intervened on step {step.idx} successfully{mode_tag}."
            self.status_msg = msg
            return True, msg
        except Exception as exc:
            msg = f"Intervention failed: {first_line(exc)}"
            self.status_msg = msg
            return False, msg

    def _fork_category(self, step: StepItem, arm_running: bool) -> str:
        """Returns the category a fork at this step acts as. Forking before a response treats tool calls as responses."""
        if arm_running and step.category.startswith("tool_call_"):
            return "model_response"
        return step.category

    def _fork_refusal(self, step: StepItem, arm_running: bool) -> str | None:
        """Explains why the selected step cannot be forked, or returns None if it can."""
        category = self._fork_category(step, arm_running)
        if category in FORKABLE_CATEGORIES:
            return None
        if category == "tool_result":
            return f"Cannot fork at a tool result; select the tool call at step {step.idx - 1}."
        if category == "tool_call_other":
            return f"Cannot fork at step {step.idx}: results of {step.detail} cannot be changed."
        return f"Cannot fork at step {step.idx}: {category} steps cannot be changed."

    def execute_action_fork(self, branch_name: str = "branch", arm_running: bool = False,
                            replace_text: str = None, clear_task_logs: bool = False) -> tuple[bool, str, str | None]:
        """Performs non-destructive fork action at the currently selected step."""
        step = self.get_selected_step()
        if not step:
            return False, "No step selected.", None

        refusal = self._fork_refusal(step, arm_running)
        if refusal:
            self.status_msg = refusal
            return False, refusal, None

        if execute_fork is None:
            return False, "execute_fork function is not available.", None

        category = self._fork_category(step, arm_running)
        intervention_kwargs = {}
        if category in rewind_mod.TOOL_INTERVENTION_CATEGORIES:
            intervention_kwargs = {"tool_name": step.detail or "tool", "replace_text": replace_text}
        elif category in rewind_mod.TASK_INTERVENTION_CATEGORIES:
            intervention_kwargs = {"detail": step.detail, "replace_text": replace_text}
        elif category == "subagent_report":
            # keep-alive changes only the text that was typed, without inventing a crash
            intervention_kwargs = {"sender_uuid": step.detail, "replace_text": replace_text, "subagent_exit": "keep-alive"}

        try:
            future_tasks = find_future_task_logs(self.base_dir, self.conv_id, step.idx, category)
            with self._captured_output():
                fork_cid = execute_fork(
                    base_dir=self.base_dir,
                    parent_conv_id=self.conv_id,
                    target_step=step.idx,
                    branch_name=branch_name,
                    dry_run=self.dry_run,
                    arm_running=(category == "model_response"),
                    category=category,
                    intervention_kwargs=intervention_kwargs,
                    task_output=(clear_log_spec(future_tasks) if clear_task_logs and future_tasks else None)
                )
            mode_tag = " [DRY-RUN]" if self.dry_run else ""
            msg = f"Forked to branch '{branch_name}' ({fork_cid[:8]}...){mode_tag}."
            self.status_msg = msg
            return True, msg, fork_cid
        except Exception as exc:
            msg = f"Fork failed: {first_line(exc)}"
            self.status_msg = msg
            return False, msg, None

    def format_inspector_lines(self, step: StepItem) -> list[tuple[str, int]]:
        """
        Formats inspector lines as (text, color_pair_id) tuples for rendering.
        Can be queried headlessly for testing output text.
        """
        lines: list[tuple[str, int]] = []
        lines.append((f"=== Step {step.idx}: {step.summary_title} ===", 1))
        lines.append((f"Category:   {step.category}", 0))
        if step.detail:
            lines.append((f"Detail:     {step.detail}", 0))
        lines.append((f"Timestamp:  {step.created_at or 'Unknown'}", 0))
        lines.append((f"Source:     {step.source}", 0))

        if step.paired_step is not None:
            lines.append((f"{step.paired_label}: Step {step.paired_step}", 3))

        if step.artifacts:
            lines.append(("", 0))
            lines.append(("--- Associated Artifacts ---", 3))
            for art in step.artifacts:
                lines.append((f"  * {art}", 0))

        if step.tool_calls:
            lines.append(("", 0))
            lines.append(("--- Tool Invocations ---", 3))
            for i, tc in enumerate(step.tool_calls):
                tname = tc.get("name", "tool")
                targs = tc.get("args", {})
                lines.append((f"[{i + 1}] Tool: {tname}", 3))
                try:
                    pretty_args = json.dumps(targs, indent=2, ensure_ascii=False)
                    for arg_line in pretty_args.splitlines():
                        lines.append((f"    {arg_line}", 0))
                except Exception:
                    lines.append((f"    Arguments: {targs}", 0))

        if step.thinking:
            lines.append(("", 0))
            lines.append(("--- Internal Thinking / Reasoning ---", 2))
            for t_line in step.thinking.strip().splitlines():
                lines.append((f"  {t_line}", 0))

        if step.display_content:
            lines.append(("", 0))
            lines.append(("--- Raw Content ---", 1))
            for c_line in step.display_content.rstrip().splitlines():
                lines.append((f"  {c_line}", 0))

        lines.append(("", 0))
        lines.append(("--- Action Hints ---", 2))
        if step.category == "prompt":
            lines.append(("  Press [R] to rewind before this prompt.", 0))
        elif step.category == "model_response":
            lines.append(("  Press [B] or [R] to rewind and regenerate from prior context.", 0))
        elif step.category.startswith("tool_call_"):
            lines.append(("  Press [B] to rewind before this tool decision (regenerate).", 0))
            lines.append(("  Press [I] to intervene and replace observation / simulate error.", 0))
        else:
            lines.append(("  Press [R] to rewind to this step.", 0))

        return lines

    # ----------------------------------------------------------------------
    # Curses Render Loop
    # ----------------------------------------------------------------------

    def header_line(self, width: int) -> str:
        """Builds the header row to fill the window exactly. The dry-run tag comes first so it can never be cut off."""
        dry_tag = " [DRY-RUN ON]" if self.dry_run else ""
        left = f"{dry_tag} AGY_REWIND: {self.conv_id[:18]}... | Status: {self.loader.status} "
        right = f" Total: {len(self.loader.steps)} turns "
        return (left + " " * max(0, width - len(left) - len(right)) + right)[:width]

    def run(self, stdscr):
        """Main curses interactive event loop."""
        import curses

        self.stdscr = stdscr
        curses.curs_set(0)
        stdscr.keypad(True)
        curses.use_default_colors()

        # Initialize color pairs
        if curses.has_colors():
            curses.init_pair(1, curses.COLOR_CYAN, -1)     # Prompts / Title
            curses.init_pair(2, curses.COLOR_BLUE, -1)     # Reasoning / Meta
            curses.init_pair(3, curses.COLOR_YELLOW, -1)   # Tool calls
            curses.init_pair(4, curses.COLOR_GREEN, -1)    # Observations / Active
            curses.init_pair(5, curses.COLOR_RED, -1)      # Cutoff / Errors
            curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # Subagent / Task

        while self.is_running:
            max_y, max_x = stdscr.getmaxyx()
            if max_y < 12 or max_x < 60:
                stdscr.clear()
                msg = "Terminal window too small (minimum 60x12 required)."
                stdscr.addstr(max_y // 2, max(0, (max_x - len(msg)) // 2), msg[:max_x - 1])
                stdscr.refresh()
                ch = stdscr.getch()
                if ch in (ord('q'), ord('Q')):
                    break
                continue

            stdscr.erase()

            # 1. Header (lines 0-1)
            stdscr.attron(curses.A_REVERSE | curses.A_BOLD)
            stdscr.addstr(0, 0, self.header_line(max_x))
            stdscr.attroff(curses.A_REVERSE | curses.A_BOLD)

            # Dimensions
            content_y = 1
            footer_h = 2
            content_h = max_y - content_y - footer_h
            left_w = min(36, max(26, max_x // 3))
            right_w = max_x - left_w - 1

            # 2. Left Timeline Panel
            self._render_timeline(stdscr, content_y, 0, content_h, left_w)

            # Divider line
            for y in range(content_y, content_y + content_h):
                try:
                    stdscr.addch(y, left_w, curses.ACS_VLINE)
                except curses.error:
                    pass

            # 3. Right Inspector Panel
            self._render_inspector(stdscr, content_y, left_w + 1, content_h, right_w)

            # 4. Footer & Status Bar
            self._render_footer(stdscr, max_y - footer_h, 0, footer_h, max_x)

            stdscr.refresh()

            # Input handling
            try:
                ch = stdscr.getch()
            except KeyboardInterrupt:
                break

            self.handle_input_char(ch)

    def handle_input_char(self, ch: int):
        """Processes keyboard input in curses mode."""
        import curses

        if ch in (curses.KEY_UP, ord('k'), ord('K')):
            self.navigate(-1)
        elif ch in (curses.KEY_DOWN, ord('j'), ord('J')):
            self.navigate(1)
        elif ch in (curses.KEY_PPAGE,):
            self.scroll_inspector(-5)
        elif ch in (curses.KEY_NPAGE,):
            self.scroll_inspector(5)
        elif ch in (curses.KEY_HOME,):
            self.cursor = 0
            self.inspector_scroll = 0
        elif ch in (curses.KEY_END,):
            self.cursor = max(0, len(self.loader.steps) - 1)
            self.inspector_scroll = 0
        elif ch in (ord('d'), ord('D')):
            self.toggle_dry_run()
        elif ch in (ord('q'), ord('Q'), 27):  # 27 = ESC
            self.is_running = False
        elif ch in (ord('r'), ord('R')):
            self._prompt_and_rewind(arm_running=False)
        elif ch in (ord('b'), ord('B')):
            self._prompt_and_rewind(arm_running=True)
        elif ch in (ord('i'), ord('I')):
            self._prompt_and_intervene()
        elif ch in (ord('f'), ord('F')):
            self._prompt_and_fork(arm_running=False)

    def _render_timeline(self, stdscr, start_y: int, start_x: int, height: int, width: int):
        """Renders vertical scrubber timeline on left panel."""
        import curses

        steps = self.loader.steps
        if not steps:
            stdscr.addstr(start_y + 1, start_x + 1, "No steps found.")
            return

        # Adjust scroll offset so cursor is visible
        if self.cursor < self.timeline_offset:
            self.timeline_offset = self.cursor
        elif self.cursor >= self.timeline_offset + height:
            self.timeline_offset = self.cursor - height + 1

        for i in range(height):
            step_idx = self.timeline_offset + i
            if step_idx >= len(steps):
                break
            step = steps[step_idx]
            y = start_y + i

            is_selected = (step_idx == self.cursor)
            is_downstream = (step_idx > self.cursor)

            prefix = "▶" if is_selected else " "
            badge = step.badge
            preview = clean_preview(step.display_content or (step.tool_calls[0].get("name") if step.tool_calls else ""), 16)
            line_str = f"{prefix}{badge} {step.idx:2d}: {preview}"
            line_str = line_str[:width - 1].ljust(width - 1)

            # Styling
            attr = curses.A_NORMAL
            if is_selected:
                attr = curses.A_REVERSE | curses.A_BOLD
            elif is_downstream:
                attr = curses.color_pair(5) | curses.A_DIM if curses.has_colors() else curses.A_DIM
            else:
                if curses.has_colors():
                    if step.category == "prompt":
                        attr = curses.color_pair(1)
                    elif step.category == "model_response":
                        attr = curses.color_pair(2)
                    elif step.category.startswith("tool_call_"):
                        attr = curses.color_pair(3)
                    elif step.category == "tool_result":
                        attr = curses.color_pair(4)

            try:
                stdscr.addstr(y, start_x, line_str, attr)
            except curses.error:
                pass

    def _render_inspector(self, stdscr, start_y: int, start_x: int, height: int, width: int):
        """Renders detailed turn inspection on right panel."""
        import curses

        step = self.get_selected_step()
        if not step:
            return

        lines = self.format_inspector_lines(step)

        # Draw visible lines
        visible_lines = lines[self.inspector_scroll:self.inspector_scroll + height]
        for i, (text, color_id) in enumerate(visible_lines):
            y = start_y + i
            text_truncated = text[:width - 1]
            attr = curses.A_NORMAL
            if color_id > 0 and curses.has_colors():
                attr = curses.color_pair(color_id)
                if color_id in (1, 3):
                    attr |= curses.A_BOLD
            try:
                stdscr.addstr(y, start_x, text_truncated, attr)
            except curses.error:
                pass

    def _render_footer(self, stdscr, start_y: int, start_x: int, height: int, width: int):
        """Renders status and keybinding hints bar."""
        import curses

        hints = "[↑/↓] Scrub | [B] Before-Resp | [R] Rewind | [I] Intervene | [F] Fork | [D] Dry-Run | [Q] Quit"
        stdscr.attron(curses.A_BOLD)
        try:
            stdscr.addstr(start_y, start_x, hints[:width - 1])
        except curses.error:
            pass
        stdscr.attroff(curses.A_BOLD)

        # Status feedback line
        status_line = f"Status: {self.status_msg}"
        try:
            stdscr.addstr(start_y + 1, start_x, status_line[:width - 1])
        except curses.error:
            pass

    # ----------------------------------------------------------------------
    # Interactive Modal Dialogs
    # ----------------------------------------------------------------------

    def _prompt_and_rewind(self, arm_running: bool = False):
        """Confirmation dialog before executing rewind."""
        step = self.get_selected_step()
        if not step:
            return

        refusal = self._rewind_refusal(step, arm_running)
        if refusal:
            self.status_msg = refusal
            return

        label = "before Model Response" if arm_running else f"to Step {step.idx}"
        target_step = step.idx
        downstream = max(0, len(self.loader.steps) - target_step)
        prompt = f"Rewind {label}? Deletes {downstream} downstream step(s). [y/N]: "

        confirmed = self._show_confirm_dialog(prompt)
        if not confirmed:
            return

        future_tasks = find_downstream_task_completions(self.base_dir, self.conv_id, step.idx)
        if self._ask_to_clear_task_logs(future_tasks):
            success, _ = self.execute_action_rewind(arm_running=arm_running, clear_task_logs=True)
            if success:
                self._show_action_output()

    def _show_action_output(self):
        """Shows what the status bar cannot: a dry-run's preview and any notice from the tool."""
        if self.dry_run and (self.last_output or self.last_notice):
            self._show_text_dialog("Dry-run preview", self.last_output + self.last_notice)
        elif self.last_notice:
            self._show_text_dialog("Notice", self.last_notice)

    def _ask_to_clear_task_logs(self, future_tasks: list[tuple[str, int, int]]) -> bool:
        """Asks whether to empty logs holding future output. Returns False if the action should be cancelled."""
        if not future_tasks:
            return True
        if self._show_confirm_dialog(f"{len(future_tasks)} task log(s) hold future output. Clear them? [y/N]: "):
            return True
        self.status_msg = task_logs_kept_message(future_tasks)
        return False

    def _prompt_and_fork(self, arm_running: bool = False):
        """Input dialog and confirmation before executing conversation fork."""
        step = self.get_selected_step()
        if not step:
            return

        refusal = self._fork_refusal(step, arm_running)
        if refusal:
            self.status_msg = refusal
            return

        category = self._fork_category(step, arm_running)
        replace_text = None
        if category in FORK_TEXT_CATEGORIES:
            noun = "tool result" if category.startswith("tool_call_") else "message"
            replace_text = self._show_input_dialog(f"Replacement text for the whole {noun}:")
            if replace_text is None or not replace_text.strip():
                self.status_msg = "Fork cancelled: no replacement text given."
                return

        branch_name = self._show_input_dialog(f"Fork @ Step {step.idx} - Branch Name (default: 'branch'):")
        if branch_name is None:
            return
        branch_name = branch_name.strip() or "branch"

        if not self._ask_to_clear_task_logs(find_future_task_logs(self.base_dir, self.conv_id, step.idx, category)):
            return

        success, msg, new_cid = self.execute_action_fork(
            branch_name=branch_name,
            arm_running=arm_running,
            replace_text=replace_text,
            clear_task_logs=True
        )
        if success:
            self._show_action_output()
        if success and not self.dry_run and new_cid:
            switch_now = self._show_confirm_dialog(f"Forked to {new_cid[:8]}... Switch TUI to new branch? [y/N]:")
            if switch_now:
                self.conv_id = new_cid
                self.loader = SessionDataLoader(self.base_dir, self.conv_id)
                self.loader.reload()
                self.cursor = max(0, min(self.cursor, len(self.loader.steps) - 1))
                self.status_msg = f"Switched TUI to branch '{branch_name}' ({new_cid[:8]})."

    def _prompt_and_intervene(self):
        """Modal dialog prompting for replacement strings."""
        step = self.get_selected_step()
        if not step:
            return

        refusal = self._intervene_refusal(step)
        if refusal:
            self.status_msg = refusal
            return

        old_str = self._show_input_dialog("Enter substring to match & replace (--replace-old):")
        if old_str is None:
            return
        new_str = self._show_input_dialog("Enter replacement string (--replace-new):")
        if new_str is None:
            return

        if self._ask_to_clear_task_logs(find_future_task_logs(self.base_dir, self.conv_id, step.idx, step.category)):
            success, _ = self.execute_action_intervene(replace_old=old_str, replace_new=new_str, clear_task_logs=True)
            if success:
                self._show_action_output()

    def _show_confirm_dialog(self, prompt: str) -> bool:
        """Displays a simple yes/no modal popup."""
        import curses

        if not self.stdscr:
            return False

        max_y, max_x = self.stdscr.getmaxyx()
        box_w = min(max_x - 4, len(prompt) + 8)
        box_h = 5
        start_y = (max_y - box_h) // 2
        start_x = (max_x - box_w) // 2

        win = curses.newwin(box_h, box_w, start_y, start_x)
        win.box()
        win.addstr(1, 2, "Confirmation", curses.A_BOLD)
        win.addstr(2, 2, prompt[:box_w - 4])
        win.refresh()

        while True:
            ch = win.getch()
            if ch in (ord('y'), ord('Y')):
                return True
            if ch in (ord('n'), ord('N'), 27):
                return False

    def _show_text_dialog(self, title: str, lines: list[str]):
        """Shows text in a popup, one page at a time. Any key moves on."""
        import curses

        if not self.stdscr:
            return

        max_y, max_x = self.stdscr.getmaxyx()
        box_w = max(20, max_x - 4)
        box_h = max(6, max_y - 2)
        text_w = box_w - 4
        page_size = box_h - 4

        wrapped = []
        for line in lines:
            wrapped.extend(textwrap.wrap(line, text_w) or [""])

        win = curses.newwin(box_h, box_w, max(0, (max_y - box_h) // 2), max(0, (max_x - box_w) // 2))
        win.keypad(True)
        for start in range(0, len(wrapped), page_size):
            win.erase()
            win.box()
            footer = "Press any key to continue" if start + page_size < len(wrapped) else "Press any key to close"
            rows = [(1, title, curses.A_BOLD), (box_h - 2, footer, curses.A_DIM)]
            rows += [(2 + i, text, curses.A_NORMAL) for i, text in enumerate(wrapped[start:start + page_size])]
            for row, text, attribute in rows:
                try:
                    win.addstr(row, 2, text[:text_w], attribute)
                except curses.error:
                    pass
            win.refresh()
            win.getch()

    def _show_input_dialog(self, prompt: str) -> str | None:
        """Displays a modal single-line text input popup."""
        import curses

        if not self.stdscr:
            return None

        max_y, max_x = self.stdscr.getmaxyx()
        box_w = min(max_x - 4, 70)
        box_h = 5
        start_y = (max_y - box_h) // 2
        start_x = (max_x - box_w) // 2

        win = curses.newwin(box_h, box_w, start_y, start_x)
        win.keypad(True)
        curses.curs_set(1)

        input_str = ""
        rejected = False
        while True:
            win.erase()
            win.box()
            win.addstr(1, 2, prompt[:box_w - 4], curses.A_BOLD)
            # Display current input
            display_str = input_str[-(box_w - 6):]
            win.addstr(2, 2, "> " + display_str)
            if rejected:
                win.addstr(3, 2, TEXT_BOX_REFUSAL[:box_w - 4])
            win.refresh()

            ch = win.getch()
            if ch in (10, 13, curses.KEY_ENTER):  # Enter
                curses.curs_set(0)
                if rejected:
                    # Text that could not be typed must not count as blank, which means "replace everything"
                    self.status_msg = TEXT_BOX_REFUSAL
                    return None
                return input_str
            if ch == 27:  # ESC
                curses.curs_set(0)
                return None
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                input_str = input_str[:-1]
            elif 32 <= ch <= 126:
                input_str += chr(ch)
            elif 128 <= ch <= 255:
                rejected = True


def launch_tui(base_dir: str, conv_id: str, dry_run: bool = False):
    """Entry point to launch the interactive curses TUI."""
    import curses

    app = RewindTUIApp(base_dir, conv_id, dry_run=dry_run)
    curses.wrapper(app.run)
