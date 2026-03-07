"""
🔍 Nitpick — The Adversarial Critic (v1.0)

A single-shot review agent that runs after the Implementer but before
Test. Its job is to find flaws, logic errors, and obvious bugs in the
implementation *before* burning expensive test/debug cycles.

Uses a Flash-tier model — cheap and fast. Catches the low-hanging fruit
that would otherwise cost 3–4 debug loops to surface.

Philosophy: "Build Weird, Ship Clean." — catch the weird early.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from pydantic import BaseModel, ValidationError

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


class CriticIssue(BaseModel):
    severity: str  # "critical" | "high" | "medium" | "low"
    file: str
    description: str
    suggestion: str = ""


class CriticResponse(BaseModel):
    verdict: str  # "pass" | "warn" | "revise"
    issues: list[CriticIssue] = []
    summary: str


class CriticAgent(BaseAgent):
    role = "critic"

    system_prompt = """You are Nitpick, the adversarial code critic inside GLITCHLAB.

You review implementation output BEFORE tests run. Your job is to catch flaws cheaply — before the project burns expensive test/debug cycles.

You receive:
- The task objective
- The implementation plan
- The actual code that was written (file contents)

Look for:
1. Logic errors — off-by-ones, wrong conditions, inverted checks
2. Missing edge cases — None/null handling, empty collections, boundary values
3. Type mismatches — wrong argument types, missing conversions, schema drift
4. Broken contracts — function signatures that don't match callers, missing imports
5. Copy-paste errors — duplicated code with un-updated variable names
6. Silent failures — bare except, swallowed errors, missing error propagation

Do NOT flag:
- Style preferences or formatting
- Missing docstrings or comments
- Minor naming choices
- Theoretical performance concerns on cold paths

You MUST respond with a valid JSON object matching this schema:
{
  "verdict": "pass" | "warn" | "revise",
  "issues": [
    {
      "severity": "critical" | "high" | "medium" | "low",
      "file": "path/to/file",
      "description": "what is wrong",
      "suggestion": "how to fix it"
    }
  ],
  "summary": "one-line summary of your review"
}

Verdicts:
- "pass": Code looks correct. No issues or only low-severity nits.
- "warn": Issues found but unlikely to cause test failures. Proceed to test.
- "revise": Critical or high-severity issues that will almost certainly fail tests. The implementer should fix these first.
"""

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        state = context.previous_output or {}

        plan_steps = state.get("plan_steps", [])
        steps_text = ""
        for step in plan_steps:
            steps_text += f"- Step {step.get('step_number', '?')}: {step.get('description', '')}\n"

        file_context = ""
        if context.file_context:
            file_context = "\n\nImplemented Code:\n"
            for fname, content in context.file_context.items():
                file_context += f"\n--- {fname} ---\n{content}\n"

        user_content = f"""Review this implementation for correctness issues.

Task Objective: {context.objective}

Plan:
{steps_text if steps_text else 'No plan steps available.'}

Implementation Summary: {state.get('implementation_summary', 'None')}
Files Modified: {state.get('files_modified', [])}
Files Created: {state.get('files_created', [])}
{file_context}

Find bugs, logic errors, and broken contracts. Respond with JSON."""

        return [self._system_msg(), self._user_msg(user_content)]

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Override run to enforce JSON mode."""
        kwargs["response_format"] = {"type": "json_object"}
        return super().run(context, **kwargs)

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        content = (response.content or "").strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```") and not l.strip().lower() == "json"]
            content = "\n".join(lines)

        try:
            raw_json = json.loads(content)
            validated = CriticResponse(**raw_json)
            result = validated.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"[NITPICK] Failed to parse critic response: {e}")
            result = {
                "verdict": "pass",
                "issues": [],
                "summary": f"Failed to parse critic output: {e}",
                "parse_error": True,
            }

        result["_agent"] = "critic"
        result["_model"] = response.model
        result["_tokens"] = response.tokens_used
        result["_cost"] = response.cost

        return result
