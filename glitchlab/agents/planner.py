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

PLANNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Reason about the task, analyze what you've read, and draft your plan before submitting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string", "description": "Your analysis of the task and codebase"},
                    "draft_plan": {"type": "string", "description": "Your draft plan outline"}
                },
                "required": ["analysis"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_function",
            "description": "Read the complete body of a function or method. Use this to verify insertion points before including them in your plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Function or method name"},
                    "file": {"type": "string", "description": "Optional file path to restrict search"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_class",
            "description": "Get a class outline showing method signatures with bodies collapsed. Use to understand class structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {"type": "string"},
                    "file": {"type": "string", "description": "Optional file path"}
                },
                "required": ["class_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_grep",
            "description": "Search the codebase for a pattern. Use to find where things are defined or used.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "file_type": {"type": "string", "default": "*.py"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": "Submit your final execution plan as JSON. This ends the planning loop. The JSON must conform to the ExecutionPlan schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_json": {"type": "string", "description": "The complete execution plan as a JSON string"}
                },
                "required": ["plan_json"]
            }
        }
    }
]


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
15. You have tools to explore the codebase before submitting your plan. Use get_function to read the exact functions you plan to modify. Use get_class to understand class structures. Use search_grep to find where things are defined. Your code_hints MUST reference exact function names and line numbers from your exploration — not guesses.
16. When you are confident in your plan, call submit_plan with the complete JSON. Do not submit a plan without first reading the target functions.
"""

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Execute the planner in an agentic loop with read-only tools."""
        import subprocess
        from pathlib import Path

        messages = self.build_messages(context)
        workspace_dir = Path(context.working_dir)
        symbol_index = context.extra.get("symbol_index")

        max_steps = 5

        for step in range(max_steps):
            step_kwargs = dict(kwargs)
            # Remove response_format — can't use JSON mode with tools
            step_kwargs.pop("response_format", None)

            if step == 0:
                step_kwargs["tool_choice"] = {"type": "function", "function": {"name": "think"}}

            # Circuit breaker: force submission on final step
            if step == max_steps - 1:
                messages.append({
                    "role": "user",
                    "content": "FINAL STEP. You MUST call submit_plan now with your best plan. No more reading or thinking.",
                })
                step_kwargs["tool_choice"] = {"type": "function", "function": {"name": "submit_plan"}}

            response = self.router.complete(
                role=self.role,
                messages=messages,
                tools=PLANNER_TOOLS,
                **step_kwargs
            )

            # Append assistant message
            assist_msg: dict[str, Any] = {"role": "assistant"}
            if response.content:
                assist_msg["content"] = response.content
            if response.tool_calls:
                assist_msg["tool_calls"] = [
                    tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
                    for tc in response.tool_calls
                ]
            messages.append(assist_msg)

            if not response.tool_calls:
                # If the model returns plain text, try to parse it as JSON (fallback)
                if response.content:
                    return self.parse_response(response, context)
                messages.append({
                    "role": "user",
                    "content": "Use your tools to explore the codebase, then call submit_plan with your JSON plan.",
                })
                continue

            for tool_call in response.tool_calls:
                tc_id = tool_call.id
                tc_name = tool_call.function.name
                try:
                    tc_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": "Invalid JSON."})
                    continue

                if tc_name == "think":
                    messages.append({
                        "role": "tool", "tool_call_id": tc_id, "name": tc_name,
                        "content": "Analysis noted. Explore the codebase or submit your plan.",
                    })

                elif tc_name == "get_function":
                    symbol = tc_args.get("symbol")
                    file_path = tc_args.get("file")
                    if symbol_index:
                        func_data = symbol_index.get_function_body(symbol, file_path)
                        if func_data:
                            res = (
                                f"Function '{symbol}' in {func_data['file']} "
                                f"(Lines {func_data['line_start']}-{func_data['line_end']}):\n\n"
                                f"{func_data['body']}"
                            )
                        else:
                            res = f"Function '{symbol}' not found."
                    else:
                        res = "AST parser unavailable."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "get_class":
                    class_name = tc_args.get("class_name")
                    file_path = tc_args.get("file")
                    if symbol_index:
                        class_data = symbol_index.get_class_outline(class_name, file_path)
                        if class_data:
                            res = (
                                f"Class '{class_name}' in {class_data['file']} "
                                f"(Lines {class_data['line_start']}-{class_data['line_end']}):\n\n"
                                f"{class_data['outline']}"
                            )
                        else:
                            res = f"Class '{class_name}' not found."
                    else:
                        res = "AST parser unavailable."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "search_grep":
                    pattern = tc_args.get("pattern")
                    file_type = tc_args.get("file_type", "*.py")
                    try:
                        cmd = [
                            "grep", "-rn", f"--include={file_type}",
                            "--exclude-dir=.glitchlab", "--exclude-dir=__pycache__",
                            "--exclude-dir=.git", pattern, ".",
                        ]
                        proc = subprocess.run(cmd, cwd=workspace_dir, capture_output=True, text=True, timeout=10)
                        lines = proc.stdout.splitlines()
                        res = "\n".join(lines[:30]) if lines else "No matches."
                    except Exception as e:
                        res = f"Search failed: {e}"
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "submit_plan":
                    plan_json = tc_args.get("plan_json", "")
                    fake_response = RouterResponse(
                        content=plan_json,
                        model=response.model,
                        tokens_used=response.tokens_used,
                        cost=response.cost,
                        latency_ms=response.latency_ms,
                    )
                    return self.parse_response(fake_response, context)

        # Fallback: loop exhausted without submit_plan
        logger.warning("[ZAP] Planner loop exhausted without calling submit_plan.")
        return {
            "steps": [],
            "files_likely_affected": [],
            "requires_core_change": False,
            "risk_level": "high",
            "risk_notes": "Planner failed to produce a plan within step limit",
            "test_strategy": [],
            "estimated_complexity": "unknown",
            "parse_error": True,
        }

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
            compact_result = prelude.compact(max_tokens=400)
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