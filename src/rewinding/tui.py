"""
Interactive Terminal UI (TUI) Visual Scrubber for AGY_rewind.
Provides keyboard-driven step scrubbing, untruncated turn inspection,
and direct one-key rewinding and counterfactual intervention.
"""

from dataclasses import dataclass, field
import json
import os
import re
import sqlite3
import sys
from typing import Any

try:
    import rewinding.Antigravity_Rewind as rewind_mod
except ModuleNotFoundError:
    import Antigravity_Rewind as rewind_mod

detect_step_info = rewind_mod.detect_step_info
execute_rewind = rewind_mod.execute_rewind
execute_tool_intervene = rewind_mod.execute_tool_intervene
clean_preview = rewind_mod.clean_preview
STEP_TYPE_USER_INPUT = rewind_mod.STEP_TYPE_USER_INPUT
STEP_TYPE_MODEL_RESPONSE = rewind_mod.STEP_TYPE_MODEL_RESPONSE


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

    def execute_action_rewind(self, arm_running: bool = False) -> tuple[bool, str]:
        """Performs rewind action on the currently selected step."""
        step = self.get_selected_step()
        if not step:
            return False, "No step selected."

        try:
            execute_rewind(
                self.base_dir,
                self.conv_id,
                target_step=step.idx,
                dry_run=self.dry_run,
                arm_running=arm_running
            )
            if not self.dry_run:
                self.loader.reload()
                self.cursor = max(0, min(self.cursor, len(self.loader.steps) - 1))
            action_name = "Rewound before response" if arm_running else f"Rewound to step {step.idx}"
            mode_tag = " [DRY-RUN]" if self.dry_run else ""
            msg = f"{action_name} successfully{mode_tag}."
            self.status_msg = msg
            return True, msg
        except Exception as exc:
            msg = f"Rewind failed: {exc}"
            self.status_msg = msg
            return False, msg

    def execute_action_intervene(self, replace_old: str = None, replace_new: str = None,
                                  replace_text: str = None, simulate_error: str = None) -> tuple[bool, str]:
        """Performs intervention action on the currently selected step."""
        step = self.get_selected_step()
        if not step:
            return False, "No step selected."

        if not step.category.startswith("tool_call_") and step.category not in ("task_notification", "subagent_report"):
            msg = f"Step {step.idx} ({step.category}) cannot be intervened upon."
            self.status_msg = msg
            return False, msg

        try:
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
            if not self.dry_run:
                self.loader.reload()
                self.cursor = max(0, min(self.cursor, len(self.loader.steps) - 1))
            mode_tag = " [DRY-RUN]" if self.dry_run else ""
            msg = f"Intervened on step {step.idx} successfully{mode_tag}."
            self.status_msg = msg
            return True, msg
        except Exception as exc:
            msg = f"Intervention failed: {exc}"
            self.status_msg = msg
            return False, msg

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
            dry_tag = " [DRY-RUN ON]" if self.dry_run else ""
            header_left = f" AGY_REWIND: {self.conv_id[:18]}... | Status: {self.loader.status}{dry_tag} "
            header_right = f" Total: {len(self.loader.steps)} turns "
            stdscr.attron(curses.A_REVERSE | curses.A_BOLD)
            stdscr.addstr(0, 0, header_left)
            space_len = max(0, max_x - len(header_left) - len(header_right))
            stdscr.addstr(0, len(header_left), " " * space_len + header_right)
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

        hints = "[↑/↓] Scrub | [B] Before-Resp | [R] Rewind | [I] Intervene | [D] Dry-Run | [Q] Quit"
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

        label = "before Model Response" if arm_running else f"to Step {step.idx}"
        target_step = step.idx
        downstream = max(0, len(self.loader.steps) - target_step)
        prompt = f"Rewind {label}? Deletes {downstream} downstream step(s). [y/N]: "

        confirmed = self._show_confirm_dialog(prompt)
        if confirmed:
            self.execute_action_rewind(arm_running=arm_running)

    def _prompt_and_intervene(self):
        """Modal dialog prompting for replacement strings."""
        step = self.get_selected_step()
        if not step:
            return

        if not step.category.startswith("tool_call_") and step.category not in ("task_notification", "subagent_report"):
            self.status_msg = f"Step {step.idx} is not an intervenable tool or notification."
            return

        old_str = self._show_input_dialog("Enter substring to match & replace (--replace-old):")
        if old_str is None:
            return
        new_str = self._show_input_dialog("Enter replacement string (--replace-new):")
        if new_str is None:
            return

        self.execute_action_intervene(replace_old=old_str, replace_new=new_str)

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
        while True:
            win.erase()
            win.box()
            win.addstr(1, 2, prompt[:box_w - 4], curses.A_BOLD)
            # Display current input
            display_str = input_str[-(box_w - 6):]
            win.addstr(2, 2, "> " + display_str)
            win.refresh()

            ch = win.getch()
            if ch in (10, 13, curses.KEY_ENTER):  # Enter
                curses.curs_set(0)
                return input_str
            if ch == 27:  # ESC
                curses.curs_set(0)
                return None
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                input_str = input_str[:-1]
            elif 32 <= ch <= 126:
                input_str += chr(ch)


def launch_tui(base_dir: str, conv_id: str, dry_run: bool = False):
    """Entry point to launch the interactive curses TUI."""
    import curses

    app = RewindTUIApp(base_dir, conv_id, dry_run=dry_run)
    curses.wrapper(app.run)
