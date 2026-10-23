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
        kickstart_idx = None
        matched_token = None
        matched_role = None

        # Search backwards across all turns in contents
        for i in range(len(contents) - 1, -1, -1):
            turn = contents[i]
            role = str(turn.get("role", "")).lower()
            parts = turn.get("parts", [])

            for part in parts:
                text = ""
                if isinstance(part, dict):
                    text = part.get("text", "")
                elif isinstance(part, str):
                    text = part

                for token in active_tokens:
                    if token and token in text:
                        kickstart_idx = i
                        matched_token = token
                        matched_role = role
                        break
                if kickstart_idx is not None:
                    break
            if kickstart_idx is not None:
                break

        if kickstart_idx is not None:
            # Check if the kickstart turn has other non-kickstart parts (e.g. SYSTEM_MESSAGE)
            target_turn = contents[kickstart_idx]
            original_parts = target_turn.get("parts", [])

            def is_kickstart_part(p):
                t = p.get("text", "") if isinstance(p, dict) else (p if isinstance(p, str) else "")
                return any(token and token in t for token in active_tokens)

            preserved_parts = [p for p in original_parts if not is_kickstart_part(p)]

            removed_indices = []
            if preserved_parts:
                target_turn["parts"] = preserved_parts
            else:
                removed_indices.append(kickstart_idx)

            next_idx = kickstart_idx + 1
            while next_idx < len(contents):
                next_turn = contents[next_idx]
                next_role = str(next_turn.get("role", "")).lower()
                next_parts = next_turn.get("parts", [])

                if next_role == "model":
                    break
                if any(isinstance(p, dict) and ("functionResponse" in p or "functionCall" in p) for p in next_parts):
                    break

                is_system_notice = False
                for p in next_parts:
                    t = p.get("text", "") if isinstance(p, dict) else (p if isinstance(p, str) else "")
                    if any(marker in t for marker in ["<SYSTEM_MESSAGE>", "[Notice]", "[Message] timestamp="]):
                        is_system_notice = True
                        break

                if is_system_notice:
                    removed_indices.append(next_idx)
                    next_idx += 1
                else:
                    break

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
                f"[*] Stripped kickstart turn! Detected token '{matched_token}' at turn index {kickstart_idx} (role: '{matched_role}'). "
                f"Removed turn indices {removed_indices}. Outbound payload now ends at '{trailing_role}' "
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
