"""
🐛 Reroute — The Debugger

Only appears when things break.
Laser-focused. No feature creep.

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

    system_prompt = """You are Reroute, the debug engine inside GLITCHLAB.

You are invoked ONLY when tests fail or builds break.
Your job is to produce a MINIMAL fix. Nothing more.

You MUST respond with valid JSON only. No markdown wrapping.

Output schema:
{
  "diagnosis": "What went wrong and why",
  "root_cause": "The specific root cause",
  "fix": {
    "changes": [
      {
        "file": "path/to/file",
        "action": "modify|create",
        "content": "The COMPLETE updated file content. ALWAYS provide this field.",
        "description": "what this fixes"
      }
    ]
  },
  "confidence": "high|medium|low",
  "should_retry": true,
  "notes": "Any additional context"
}

CRITICAL RULES:
- Fix the EXACT failure. Nothing else.
- Do not refactor. Do not improve. Do not add features.
- If you cannot fix it with confidence, set should_retry=false.
- ALWAYS provide the COMPLETE file content in the 'content' field.
- Do NOT use unified diffs or patches. Provide full file content only.
- Do NOT wrap your JSON response in markdown code fences.
- Analyze the error output carefully before proposing changes.
- If the error is an import error or missing class, check that all referenced modules exist.
- If the error is a syntax error, provide the corrected full file.
"""

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        error_output = context.extra.get("error_output", "No error output provided")
        test_command = context.extra.get("test_command", "unknown")
        attempt = context.extra.get("attempt", 1)
        previous_fixes = context.extra.get("previous_fixes", [])
        patch_strategy = context.extra.get("patch_strategy", "")

        prev_fixes_text = ""
        if previous_fixes:
            prev_fixes_text = "\n\nPrevious fix attempts that did NOT work:\n"
            for i, fix in enumerate(previous_fixes, 1):
                diag = fix.get("diagnosis", "unknown")
                apply_result = fix.get("_apply_result", [])
                prev_fixes_text += f"\nAttempt {i}: {diag}\n"
                if apply_result:
                    prev_fixes_text += f"  Apply result: {apply_result}\n"

        file_context = ""
        if context.file_context:
            file_context = "\n\nCurrent file contents (use these as the base for your fix):\n"
            for fname, content in context.file_context.items():
                file_context += f"\n--- {fname} ---\n{content}\n"

        strategy_note = ""
        if patch_strategy:
            strategy_note = f"\n\n⚠️ {patch_strategy}\n"

        user_content = f"""Test/build failure detected.

Task: {context.objective}
Task ID: {context.task_id}
Fix attempt: {attempt}

Command that failed: {test_command}

Error output:
```
{error_output}
```
{strategy_note}
{prev_fixes_text}
{file_context}

Diagnose the failure and produce a minimal fix as JSON.
Remember: provide COMPLETE file content in the 'content' field. Do NOT use patches."""

        return [self._system_msg(), self._user_msg(user_content)]

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        """Parse debug output from Reroute."""
        content = response.content.strip()

        # Log raw response for debugging patch issues
        logger.debug(f"[REROUTE] Raw response ({len(content)} chars):\n{content[:2000]}")

        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"[REROUTE] Failed to parse debug JSON: {e}")
            logger.debug(f"[REROUTE] Content that failed to parse:\n{content[:1000]}")
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    result = {
                        "diagnosis": "Failed to parse debugger output",
                        "root_cause": str(e),
                        "fix": {"changes": []},
                        "confidence": "low",
                        "should_retry": False,
                        "parse_error": True,
                    }
            else:
                result = {
                    "diagnosis": "Failed to parse debugger output",
                    "root_cause": str(e),
                    "fix": {"changes": []},
                    "confidence": "low",
                    "should_retry": False,
                    "parse_error": True,
                }

        # Post-process: if any change has 'patch' but no 'content', log a warning
        for change in result.get("fix", {}).get("changes", []):
            if change.get("patch") and not change.get("content"):
                logger.warning(
                    f"[REROUTE] Change for {change.get('file', '?')} has patch but no content. "
                    "This will likely fail to apply."
                )

        result["_agent"] = "debugger"
        result["_model"] = response.model
        result["_tokens"] = response.tokens_used
        result["_cost"] = response.cost

        logger.info(
            f"[REROUTE] Diagnosis: {result.get('diagnosis', '?')[:80]} — "
            f"confidence={result.get('confidence', '?')}, "
            f"retry={result.get('should_retry', False)}"
        )

        return result