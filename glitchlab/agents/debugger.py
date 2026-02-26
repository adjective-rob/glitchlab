"""
🐛 Reroute — The Debugger (v2.2)

Only appears when things break. Laser-focused.
Hardened against truncated / malformed LLM JSON responses.

Energy: quiet gremlin that only appears when things break.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


class DebuggerAgent(BaseAgent):
    role = "debugger"

    # Immutable system contract
    system_prompt = """You are Reroute, the debug engine inside GLITCHLAB.

You are invoked ONLY when tests fail or builds break.
Your job is to produce a MINIMAL, surgically precise fix.

You MUST respond with valid JSON only. No markdown wrapping.

Output schema:
{
  "diagnosis": "Short summary of what failed",
  "root_cause": "The specific line or logic error",
  "fix": {
    "changes": [
      {
        "file": "path/to/file",
        "action": "modify",
        "surgical_blocks": [
          {
            "search": "The EXACT lines to find in the original file, including 2-3 lines of unchanged context above and below to ensure uniqueness.",
            "replace": "The new lines that will replace the search block."
          }
        ],
        "description": "Fixes the specific error"
      }
    ]
  },
  "confidence": "high|medium|low",
  "should_retry": true
}

CRITICAL RULES:
1. Fix the EXACT failure. Do not refactor unrelated code.
2. YOU MUST USE `surgical_blocks` FOR MODIFICATIONS. NEVER output the full file content.
3. The `search` string must EXACTLY match the original file character-for-character, including whitespace and indentation.
4. If tokens run out, prioritize completing the JSON structure.
5. Do NOT wrap output in markdown.
6. If Exit Code 5 (No tests collected), check for missing __init__.py.
"""

    # ------------------------------------------------------------------ #
    # Prompt Construction
    # ------------------------------------------------------------------ #

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        """Construct prompt with capped error context to preserve token headroom."""

        state = context.previous_output or {}

        raw_error = context.extra.get("error_output", "") or ""
        error_output = raw_error[:1000]  # Hard cap for token safety

        test_command = context.extra.get("test_command", "unknown")
        attempt = state.get("debug_attempts", 1)
        patch_strategy = context.extra.get("patch_strategy", "")

        # Previous attempts (limit to last 2)
        previous_fixes = state.get("previous_fixes", [])
        prev_fixes_text = ""
        if previous_fixes:
            prev_fixes_text = "\n\nFAILED PREVIOUS ATTEMPTS:\n"
            for i, fix in enumerate(previous_fixes[-2:], 1):
                diag = fix.get("diagnosis", "unknown")
                prev_fixes_text += f"Attempt {i}: {diag}\n"

        # File context
        file_context = ""
        if context.file_context:
            file_context = "\n\nFILES FOR CONTEXT (Use as base for fix):\n"
            for fname, content in context.file_context.items():
                if fname.startswith("[dep] "):
                    file_context += f"\n[SIGNATURES] {fname[6:]}:\n{content}\n"
                else:
                    file_context += f"\n--- {fname} ---\n{content}\n"

        user_content = f"""FAILURE DETECTED (Attempt {attempt})

Objective: {context.objective}
Mode: {state.get('mode', 'evolution')}
Modified files: {state.get('files_modified', [])}

FAILED COMMAND: {test_command}

ERROR LOG (TRUNCATED):
{error_output}

{f"⚠️ STRATEGY: {patch_strategy}" if patch_strategy else ""}
{prev_fixes_text}
{file_context}

Produce the minimal JSON fix using strict surgical_blocks."""

        return [self._system_msg(), self._user_msg(user_content)]

    # ------------------------------------------------------------------ #
    # Response Parsing
    # ------------------------------------------------------------------ #

    def parse_response(
        self,
        response: RouterResponse,
        context: AgentContext
    ) -> dict[str, Any]:
        """Parse debug output with resilient JSON extraction + repair."""

        content = response.content.strip()

        # Remove markdown fences if model violated rules
        content = self._strip_markdown(content)

        # Attempt direct parse
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("[REROUTE] Malformed JSON detected. Attempting recovery...")
            result = self._recover_json(content)

        # Attach metadata
        result["_agent"] = "debugger"
        result["_model"] = response.model
        result["_tokens"] = response.tokens_used

        logger.info(
            f"[REROUTE] Diagnosis: {result.get('diagnosis', 'Unknown')[:60]}... "
            f"Confidence: {result.get('confidence', 'low')}"
        )

        return result

    # ------------------------------------------------------------------ #
    # JSON Recovery Logic
    # ------------------------------------------------------------------ #

    def _strip_markdown(self, content: str) -> str:
        """Remove ``` or ```json wrappers safely."""
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        return content.strip()

    def _recover_json(self, content: str) -> dict[str, Any]:
        """
        Recover largest valid JSON object from truncated response.
        Strategy:
        1. Extract first full JSON object if embedded in noise.
        2. Attempt structural balancing.
        3. Fallback to regex extraction.
        """

        # 1. Try extracting largest {...} block
        extracted = self._extract_outer_json(content)
        if extracted:
            try:
                return json.loads(extracted)
            except Exception:
                pass

        # 2. Structural balancing
        balanced = self._balance_json(content)
        if balanced:
            try:
                return json.loads(balanced)
            except Exception:
                pass

        # 3. Regex fallback
        logger.error("[REROUTE] JSON recovery failed. Using minimal fallback.")
        diag_match = re.search(r'"diagnosis"\s*:\s*"([^"]+)"', content)

        return {
            "diagnosis": diag_match.group(1) if diag_match else "Truncated response",
            "root_cause": "JSON_TRUNCATION",
            "fix": {"changes": []},
            "confidence": "low",
            "should_retry": False,
            "parse_error": True,
        }

    def _extract_outer_json(self, text: str) -> str | None:
        """Extract first top-level JSON object using brace tracking."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        return None

    def _balance_json(self, text: str) -> str | None:
        """Attempt to balance braces and brackets in truncated JSON."""

        open_braces = text.count("{")
        close_braces = text.count("}")
        open_brackets = text.count("[")
        close_brackets = text.count("]")

        repaired = text

        # Close unterminated string if odd quotes
        if repaired.count('"') % 2 != 0:
            repaired += '"'

        # Close brackets first
        if open_brackets > close_brackets:
            repaired += "]" * (open_brackets - close_brackets)

        # Then braces
        if open_braces > close_braces:
            repaired += "}" * (open_braces - close_braces)

        return repaired if repaired != text else None