#!/usr/bin/env python3
"""
Antigravity Kickstart Interceptor

Mitmproxy addon script that intercepts outbound Google Cloud / Gemini API inference calls,
detects the kickstart starter-motor token in the trailing user turn, and silently strips it.
This ensures pure counterfactual causal downstream generation with zero prompt contamination.

Supported Tokens:
    - "Stop doing everything and standby" (default)
    - "__AGY_KICKSTART__" (default fallback)
    - Custom string via AGY_KICKSTART_TOKEN env var or ~/.kickstart_token file
"""

import json
import logging
import os
import sys

DEFAULT_KICKSTART_TOKEN = "__AGY_KICKSTART__"

# How a system message starts. Real prompts can mention these words, so only the start of the text counts.
SYSTEM_NOTICE_PREFIXES = ("The following is a <SYSTEM_MESSAGE>", "<SYSTEM_MESSAGE>", "[Notice]", "[Message] timestamp=")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("interceptor")


class KickstartStripper:
    def __init__(self):
        env_token = os.environ.get("AGY_KICKSTART_TOKEN", DEFAULT_KICKSTART_TOKEN)
        self.default_token = env_token
        logger.info(f"Initialized KickstartStripper with active token: '{self.default_token}'")

    def get_active_tokens(self) -> list[str]:
        tokens = [self.default_token]
        token_file = os.path.expanduser("~/.kickstart_token")
        if os.path.isfile(token_file):
            try:
                with open(token_file, "r", encoding="utf-8") as f:
                    file_token = f.read().strip()
                    if file_token and file_token not in tokens:
                        tokens.insert(0, file_token)
            except Exception as e:
                logger.warning(f"Failed to read {token_file}: {e}")
        return tokens

    @staticmethod
    def locate_contents(payload: dict):
        """Finds the contents array inside the JSON payload regardless of wrapping."""
        if not isinstance(payload, dict):
            return None, None

        if isinstance(payload.get("contents"), list):
            return payload, "contents"

        # Check common wrapper envelopes used by Cloud Code / Vertex endpoints
        for wrapper_key in ("request", "modelRequest", "generateContentRequest", "clientRequest"):
            sub = payload.get(wrapper_key)
            if isinstance(sub, dict) and isinstance(sub.get("contents"), list):
                return sub, "contents"

        return None, None

    def request(self, flow) -> None:
        if flow.request.method != "POST":
            return

        path = flow.request.path.lower()
        if "generatecontent" not in path:
            return

        logger.info(f"[INTERCEPTOR] Inspecting target request: {flow.request.method} {flow.request.path}")

        # Decode body
        try:
            body_text = flow.request.get_text()
            if not body_text:
                logger.warning("[INTERCEPTOR] Request body is empty, skipping.")
                return
            payload = json.loads(body_text)
        except Exception as e:
            logger.warning(f"[INTERCEPTOR] Failed to decode/parse request body as JSON: {e}")
            return

        parent_dict, contents_key = self.locate_contents(payload)
        if parent_dict is None:
            logger.warning(
                f"[INTERCEPTOR] Could not find 'contents' array in payload. Top-level keys: {list(payload.keys())}"
            )
            return

        contents = parent_dict[contents_key]
        if not contents:
            logger.info("[INTERCEPTOR] 'contents' array is empty, skipping.")
            return

        active_tokens = self.get_active_tokens()

        def part_text(part) -> str:
            return part.get("text", "") if isinstance(part, dict) else (part if isinstance(part, str) else "")

        def is_kickstart_part(part) -> bool:
            return any(token and token in part_text(part) for token in active_tokens)

        # A model turn quoting the token is the model's own words, not a kickstart. An earlier kickstart
        # stays in the history that later requests carry, so every one is removed, not just the newest.
        kickstart_indices = [
            i for i, turn in enumerate(contents)
            if str(turn.get("role", "")).lower() != "model" and any(is_kickstart_part(p) for p in turn.get("parts", []))
        ]

        if kickstart_indices:
            removed_indices = set()
            for kickstart_idx in kickstart_indices:
                # The kickstart turn may have other parts (e.g. SYSTEM_MESSAGE), which are kept
                target_turn = contents[kickstart_idx]
                preserved_parts = [p for p in target_turn.get("parts", []) if not is_kickstart_part(p)]
                if preserved_parts:
                    target_turn["parts"] = preserved_parts
                else:
                    removed_indices.add(kickstart_idx)

                next_idx = kickstart_idx + 1
                while next_idx < len(contents):
                    next_turn = contents[next_idx]
                    next_role = str(next_turn.get("role", "")).lower()
                    next_parts = next_turn.get("parts", [])

                    if next_role == "model":
                        break
                    if any(isinstance(p, dict) and ("functionResponse" in p or "functionCall" in p) for p in next_parts):
                        break
                    if not any(part_text(p).lstrip().startswith(SYSTEM_NOTICE_PREFIXES) for p in next_parts):
                        break
                    removed_indices.add(next_idx)
                    next_idx += 1

            if removed_indices:
                contents = [turn for idx, turn in enumerate(contents) if idx not in removed_indices]
                parent_dict[contents_key] = contents

            # Determine what the trailing turn is now
            trailing_role = contents[-1].get("role", "none") if contents else "empty"
            has_tool_res = False
            if contents and "parts" in contents[-1]:
                has_tool_res = any(
                    "functionResponse" in p
                    for p in contents[-1]["parts"]
                    if isinstance(p, dict)
                )

            logger.info(
                f"[*] Stripped kickstart turn(s) at indices {kickstart_indices}. "
                f"Removed turn indices {sorted(removed_indices)}. Outbound payload now ends at '{trailing_role}' "
                f"(contains functionResponse: {has_tool_res}). Total turns preserved: {len(contents)}"
            )

            # Re-serialize modified JSON back to the flow request
            flow.request.set_text(json.dumps(payload))
        else:
            last_turn = contents[-1] if contents else {}
            last_role = str(last_turn.get("role", "")).lower()
            last_parts = last_turn.get("parts", [])
            last_snippet = ""
            for p in last_parts:
                t = p.get("text", "") if isinstance(p, dict) else (p if isinstance(p, str) else "")
                if t:
                    last_snippet = t[:80].replace("\n", " ")
                    break

            logger.info(
                f"[INTERCEPTOR] No kickstart token matched in any turn (total turns={len(contents)}, "
                f"last role='{last_role}', snippet='{last_snippet}'). Forwarding as-is."
            )


# Mitmproxy addon registration
addons = [KickstartStripper()]

if __name__ == "__main__":
    print("Antigravity Kickstart Interceptor Addon")
    print(f"Default token: {DEFAULT_KICKSTART_TOKEN}")
    print("\nRun with mitmproxy / mitmdump:")
    print("  mitmdump -s interceptor.py -p 8080")
