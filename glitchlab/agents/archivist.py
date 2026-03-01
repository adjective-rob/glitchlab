"""
📚 Archivist Nova — Docs + ADR Writer

Captures design decisions after successful PRs.
Updates architecture notes.
Keeps future-you sane.

Energy: library robot with LED eyes.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


class ArchivistAgent(BaseAgent):
    role = "archivist"

    system_prompt = """You are Archivist Nova, the documentation engine inside GLITCHLAB.

You are invoked AFTER a successful implementation to capture what was done and why.
You operate EXCLUSIVELY in a tool-calling loop. You do not provide raw JSON in your text responses.

Rules:
1. MANDATORY START: You MUST use the `think` tool first to plan your documentation strategy.
2. SURGICAL UPDATES: If updating an existing file (like README.md), you MUST use `read_file` to see the current content, then use `replace_in_file` to make surgical updates. 
3. NO TRUNCATION: Never use `write_file` on large existing documents. You will be penalized for deleting existing documentation.
4. ADR POLICY: Write ADRs for any change that affects architecture, public API, or introduces new patterns.
5. FINALIZATION: When all files are updated via tools, call the `done` tool to submit your final architectural notes and ADR data.
"""

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        # v2: previous_output is TaskState.to_agent_summary("archivist")
        # Contains: task_id, objective, mode, risk_level,
        #           plan_steps, files_modified, implementation_summary, version_bump
        state = context.previous_output

        # Build changes text from structured state
        files_modified = state.get("files_modified", [])
        files_text = "\n".join(f"- {f}" for f in files_modified) if files_modified else "- None"

        # Plan context from structured state
        plan_steps = state.get("plan_steps", [])
        steps_text = ""
        for step in plan_steps:
            steps_text += (
                f"\n- Step {step.get('step_number', '?')}: "
                f"{step.get('description', 'no description')}"
            )

        user_content = f"""A task has been successfully completed. Document it.

Task: {context.objective}
Task ID: {context.task_id}
Mode: {state.get('mode', 'evolution')}
Risk level: {state.get('risk_level', 'unknown')}
Version bump: {state.get('version_bump', 'unknown')}

Implementation summary: {state.get('implementation_summary', 'No summary')}

Plan steps:
{steps_text}

Files modified:
{files_text}

Existing docs in repo:
{chr(10).join(f'- {f}' for f in context.extra.get('existing_docs', []))}

Produce documentation artifacts as JSON. Set should_write_adr=false for trivial changes."""

        return [self._system_msg(), self._user_msg(user_content)]

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        """Parse documentation output from Nova."""
        content = response.content.strip()

        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"[NOVA] Failed to parse archivist JSON: {e}")
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    result = {
                        "adr": None,
                        "doc_updates": [],
                        "should_write_adr": False,
                        "architecture_notes": "Documentation generation failed",
                        "parse_error": True,
                    }
            else:
                result = {
                    "adr": None,
                    "doc_updates": [],
                    "should_write_adr": False,
                    "architecture_notes": "Documentation generation failed",
                    "parse_error": True,
                }

        result["_agent"] = "archivist"
        result["_model"] = response.model
        result["_tokens"] = response.tokens_used
        result["_cost"] = response.cost

        should_adr = result.get("should_write_adr", False)
        n_docs = len(result.get("doc_updates", []))
        logger.info(f"[NOVA] ADR: {should_adr} | Doc updates: {n_docs}")

        return result