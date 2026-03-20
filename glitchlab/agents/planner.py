"""
🧠 Professor Zap — The Planner

Breaks down tasks into execution steps.
Identifies risks, maps impacted files, decides scope.
Never writes code. Only plans.

Energy: manic genius with whiteboard chaos.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


# ---------------------------------------------------------------------------
# Strict Output Schemas
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    step_number: int
    description: str
    files: list[str] = Field(min_length=1, description="Must contain at least one valid file path")
    action: Literal["modify", "create", "delete"]
    do_not_touch: list[str] = Field(
        default_factory=list,
        description="Files or functions adjacent to this change that must NOT be modified",
    )
    code_hint: str = Field(
        default="",
        description="Pseudocode or code sketch showing the shape of the change. Not a full implementation.",
    )


class ExecutionPlan(BaseModel):
    steps: list[PlanStep]
    files_likely_affected: list[str]
    do_not_touch: list[str] = Field(
        default_factory=list,
        description="Files, functions, or code regions that are adjacent to the change but must remain untouched",
    )
    requires_core_change: bool
    risk_level: Literal["low", "medium", "high"]
    risk_notes: str
    test_strategy: list[str]
    estimated_complexity: Literal["trivial", "small", "medium", "large"]
    dependencies_affected: bool
    public_api_changed: bool
    self_review_notes: str


# ---------------------------------------------------------------------------
# Agent Implementation
# ---------------------------------------------------------------------------

class PlannerAgent(BaseAgent):
    role = "planner"

    system_prompt = """You are Professor Zap, the planning engine inside GLITCHLAB.

Your job is to take a development task and produce a precise, actionable execution plan.

You MUST respond with a valid JSON object ONLY. No markdown, no commentary.

Output schema:
{
  "steps": [
    {
      "step_number": 1,
      "description": "What to do — be specific about the change, not 'read file X'",
      "files": ["path/to/file.py"],
      "action": "modify|create|delete",
      "do_not_touch": ["path/to/adjacent.py", "ClassName.method_name"],
      "code_hint": "Add `fast_mode: bool` parameter to the extra dict, computed from self._state.files_in_scope <= 3"
    }
  ],
  "files_likely_affected": ["path/to/file1", "path/to/file2"],
  "do_not_touch": ["path/to/untouchable.py", "SomeClass.some_method"],
  "requires_core_change": false,
  "risk_level": "low|medium|high",
  "risk_notes": "Why this risk level",
  "test_strategy": ["What tests to add or run"],
  "estimated_complexity": "trivial|small|medium|large",
  "dependencies_affected": false,
  "public_api_changed": false,
  "self_review_notes": "Verification of plan against user constraints"
}

Rules:
1. Be precise about file paths. Use the file context provided.
2. Include ALL files required to satisfy the objective.
3. Keep steps minimal. Fewer steps = fewer patch errors.
4. Flag core changes honestly — this triggers human review.
5. If the task is ambiguous, say so in risk_notes.
6. Never suggest changes outside the task scope.
7. Consider test strategy for every plan.
8. DO NOT add steps to run tests, formatters, or CLI commands. You only plan file creations, modifications, and deletions.
9. DO NOT add steps whose only purpose is reading or exploring files. The implementer has read_file, get_function, and search_grep tools for exploration. Every step MUST describe a concrete write action.
10. Every step MUST have at least one valid file path in the 'files' array.
11. Every modify or create step MUST include a code_hint showing the shape of the change (1-5 lines of pseudocode, a signature sketch, or a short code fragment). The implementer uses this as a starting point. If you skip code_hint, the implementer has to guess what you meant.
12. Every step MUST populate do_not_touch with adjacent files or functions that are related to the change but must remain unmodified. Think: what will the implementer be tempted to edit that it shouldn't? List those. If genuinely nothing is adjacent, use an empty list.
13. The top-level do_not_touch covers the entire plan. Per-step do_not_touch covers that specific step. Both MUST be populated.
14. If the task mentions constraints about what NOT to change, those MUST appear in do_not_touch.
"""

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Override run to enforce JSON mode at the API level."""
        kwargs["response_format"] = {"type": "json_object"}
        return super().run(context, **kwargs)

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        file_context = ""
        if context.file_context:
            file_context = "\n\nRelevant file contents:\n"
            for fname, content in context.file_context.items():
                file_context += f"\n--- {fname} ---\n{content}\n"

        # Prelude: inject compact project context if available
        project_context = ""
        prelude = context.extra.get("prelude")
        if prelude and hasattr(prelude, 'compact') and callable(prelude.compact):
            compact_result = prelude.compact(topic=context.objective, max_tokens=400)
            if compact_result:
                project_context = f"\n\nProject context:\n{compact_result}\n"

        user_content = f"""Task: {context.objective}

Repository: {context.repo_path}
Task ID: {context.task_id}

Constraints:
{chr(10).join(f'- {c}' for c in context.constraints) if context.constraints else '- None specified'}

Acceptance Criteria:
{chr(10).join(f'- {c}' for c in context.acceptance_criteria) if context.acceptance_criteria else '- Tests pass, clean diff'}
{file_context}{project_context}

Produce your execution plan as JSON."""

        return [self._system_msg(), self._user_msg(user_content)]

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        """Parse and rigorously validate the JSON plan from Professor Zap."""
        content = response.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```") and not l.strip().lower() == "json"]
            content = "\n".join(lines)

        try:
            raw_json = json.loads(content)

            # STRICT VALIDATION: Throws ValidationError if the LLM hallucinated keys/values/actions
            validated_plan = ExecutionPlan(**raw_json)
            plan = validated_plan.model_dump()

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"[ZAP] Failed to parse/validate plan JSON: {e}")
            logger.debug(f"[ZAP] Raw response: {content[:500]}")
            plan = {
                "steps": [],
                "files_likely_affected": [],
                "requires_core_change": False,
                "risk_level": "high",
                "risk_notes": f"Validation failed: {e}",
                "test_strategy": [],
                "estimated_complexity": "unknown",
                "parse_error": True,
                "raw_response": content[:1000],
            }

        # Quality warnings — plan parsed but may be incomplete
        if "parse_error" not in plan:
            self._warn_on_quality_gaps(plan)

        plan["_agent"] = "planner"
        plan["_model"] = response.model
        plan["_tokens"] = response.tokens_used
        plan["_cost"] = response.cost

        logger.info(
            f"[ZAP] Plan ready — "
            f"{len(plan.get('steps', []))} steps, "
            f"risk={plan.get('risk_level', '?')}, "
            f"core_change={plan.get('requires_core_change', False)}"
        )
        if "self_review_notes" in plan:
            logger.info(f"[ZAP] Self-review: {plan['self_review_notes']}")

        return plan

    @staticmethod
    def _warn_on_quality_gaps(plan: dict) -> None:
        """Log warnings when the planner skips fields that improve implementer accuracy."""
        if not plan.get("do_not_touch"):
            logger.warning("[ZAP] ⚠ Plan-level do_not_touch is empty. Implementer has no boundary guidance.")

        for step in plan.get("steps", []):
            n = step.get("step_number", "?")
            action = step.get("action", "")

            if action in ("modify", "create"):
                if not step.get("code_hint"):
                    logger.warning(f"[ZAP] ⚠ Step {n}: missing code_hint. Implementer will have to guess the change shape.")
                if not step.get("do_not_touch"):
                    logger.warning(f"[ZAP] ⚠ Step {n}: missing do_not_touch. Implementer may drift into adjacent code.")