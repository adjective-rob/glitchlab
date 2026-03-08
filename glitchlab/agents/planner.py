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

import yaml
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
    # Literal types prevent the LLM from hallucinating unsupported actions
    action: Literal["modify", "create", "delete"]


class ExecutionPlan(BaseModel):
    steps: list[PlanStep]
    files_likely_affected: list[str]
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

You MUST respond with valid YAML ONLY. No markdown, no commentary, no code fences.

Output schema:
steps:
  - step_number: 1
    description: What to do
    files:
      - path/to/file.rs
    action: modify|create|delete
files_likely_affected:
  - path/to/file1
  - path/to/file2
requires_core_change: false
risk_level: low|medium|high
risk_notes: Why this risk level
test_strategy:
  - What tests to add or run
estimated_complexity: trivial|small|medium|large
dependencies_affected: false
public_api_changed: false
self_review_notes: Verification of plan against user constraints

Rules:
- Be precise about file paths. Use the file context provided.
- Include ALL files required to satisfy the objective.
- Keep steps minimal. Fewer steps = fewer patch errors.
- Flag core changes honestly — this triggers human review.
- If the task is ambiguous, say so in risk_notes.
- Never suggest changes outside the task scope.
- Consider test strategy for every plan.
- DO NOT add steps to run tests, formatters, or CLI commands. You only plan file creations, modifications, and deletions.
- Every step MUST have at least one valid file path in the 'files' array.
"""

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Override run — YAML output does not need a special response_format."""
        return super().run(context, **kwargs)

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        task_data: dict[str, Any] = {
            "task": context.objective,
            "repository": context.repo_path,
            "task_id": context.task_id,
            "constraints": context.constraints if context.constraints else ["None specified"],
            "acceptance_criteria": context.acceptance_criteria if context.acceptance_criteria else ["Tests pass, clean diff"],
        }

        file_context = ""
        if context.file_context:
            file_context = "\n\nRelevant file contents:\n"
            for fname, content in context.file_context.items():
                file_context += f"\n--- {fname} ---\n{content}\n"

        user_content = self._yaml_block(task_data)
        if file_context:
            user_content += file_context
        user_content += "\n\nProduce your execution plan as YAML."

        return [self._system_msg(), self._user_msg(user_content)]

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        """Parse and rigorously validate the YAML plan from Professor Zap."""
        content = response.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [
                l for l in lines
                if not l.strip().startswith("```")
                and l.strip().lower() not in ("json", "yaml", "yml")
            ]
            content = "\n".join(lines)

        try:
            # Try YAML first (primary format)
            raw = yaml.safe_load(content)
            if not isinstance(raw, dict):
                raise ValueError("YAML root must be a mapping")

            # STRICT VALIDATION: Throws ValidationError if the LLM hallucinated keys/values/actions
            validated_plan = ExecutionPlan(**raw)
            plan = validated_plan.model_dump()

        except (yaml.YAMLError, ValueError, ValidationError) as e:
            # Fallback: try JSON in case the model ignores format instructions
            try:
                raw = json.loads(content)
                validated_plan = ExecutionPlan(**raw)
                plan = validated_plan.model_dump()
            except (json.JSONDecodeError, ValidationError, TypeError):
                logger.error(f"[ZAP] Failed to parse/validate plan: {e}")
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