"""
🔧 Patch — The Implementer (v2.1)

Writes code. Now supports surgical Search & Replace blocks
to prevent JSON truncation on large files.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


# ---------------------------------------------------------------------------
# Surgical Output Schemas
# ---------------------------------------------------------------------------

class SurgicalBlock(BaseModel):
    """A single Search & Replace operation."""
    search: str = Field(..., description="The EXACT snippet of code to look for.")
    replace: str = Field(..., description="The code that should replace the search block.")

class FileChange(BaseModel):
    """Schema for an individual file modification."""
    file: str
    action: Literal["modify", "create", "delete"]
    content: str | None = Field(default=None)
    surgical_blocks: list[SurgicalBlock] = Field(default_factory=list)
    description: str = ""  # Make this optional

class TestChange(BaseModel):
    file: str
    content: str
    description: str = ""  # Make this optional

class ImplementationResult(BaseModel):
    changes: list[FileChange] = Field(default_factory=list)
    tests_added: list[TestChange] = Field(default_factory=list)
    commit_message: str = "chore: implementation update"  # Give a safe default
    summary: str = "Implementation completed."


# ---------------------------------------------------------------------------
# Agent Implementation
# ---------------------------------------------------------------------------

class ImplementerAgent(BaseAgent):
    role = "implementer"

    system_prompt = """You are Patch, the surgical implementation engine.

You MUST respond with a valid JSON object. No markdown wrapping.

STRATEGY FOR LARGE FILES (>100 lines):
- Do NOT rewrite the whole file in 'content'. This causes JSON truncation.
- Instead, use 'surgical_blocks' to perform Search & Replace edits.
- Each block must contain enough unique code in 'search' to be found accurately.

STRATEGY FOR SMALL FILES (<100 lines):
- Provide the FULL file in 'content' and leave 'surgical_blocks' empty.

Output schema:
{
  "changes": [
    {
      "file": "path/to/file",
      "action": "modify",
      "surgical_blocks": [
        {
          "search": "def old_func():\\n    pass",
          "replace": "def new_func():\\n    return True"
        }
      ],
      "description": "surgical update"
    }
  ],
  "tests_added": [...],
  "commit_message": "...",
  "summary": "..."
}

CRITICAL RULES:
1. Whitespace in 'search' blocks must be EXACT.
2. If using 'surgical_blocks', leave 'content' as null.
3. For NEW files (action='create'), always use 'content'.
"""

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        kwargs["response_format"] = {"type": "json_object"}
        return super().run(context, **kwargs)

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        state = context.previous_output
        
        # Determine if we should nudge the model toward surgical edits
        files_in_scope = state.get("files_in_scope", [])
        is_large_task = len(files_in_scope) > 2 or state.get("estimated_complexity") == "high"

        steps_text = ""
        for step in state.get("plan_steps", []):
            steps_text += f"\nStep {step.get('step_number')}: {step.get('description')}\n"

        file_context = ""
        if context.file_context:
            file_context = "\n\nCurrent file contents:\n"
            for fname, content in context.file_context.items():
                file_context += f"\n--- {fname} ---\n{content}\n"

        user_content = f"""Task: {context.objective}
Plan: {steps_text}
{file_context}

IMPORTANT: If modifying large files, use 'surgical_blocks' to avoid JSON truncation."""

        return [self._system_msg(), self._user_msg(user_content)]

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        content = response.content.strip()
        
        # 1. Strip markdown code fences if the model stubbornly included them
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(l for l in lines if not l.strip().startswith("```"))
            
        try:
            raw_json = json.loads(content)
            validated_impl = ImplementationResult(**raw_json)
            result = validated_impl.model_dump()
        except Exception as e:
            # 2. ACTUALLY LOG THE ERROR so we aren't blind!
            logger.error(f"[IMPLEMENTER] Parsing failed: {e}\nContent snippet: {content[:300]}")
            result = self._fallback_result(content, str(e))

        result["_agent"] = "implementer"
        return result

    @staticmethod
    def _fallback_result(raw: str, error: str) -> dict[str, Any]:
        return {
            "changes": [],
            "tests_added": [],
            "commit_message": "fix: implementation (parse error)",
            "summary": f"Failed to parse: {error}",
            "parse_error": True,
            "raw_response": raw[:2000],
        }