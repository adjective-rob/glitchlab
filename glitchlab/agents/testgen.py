"""
🛡️ Shield — The Regression Test Generator

A single-shot agent that writes a minimal regression test for new changes.
Runs automatically after implementation if no tests were written.
"""

from __future__ import annotations

import json
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, ValidationError

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


class TestGenResponse(BaseModel):
    test_file: str
    content: str
    description: str


class TestGenAgent(BaseAgent):
    role = "testgen"

    system_prompt = """You are Shield, the regression test generator.

Given a summary of code changes and the modified files, write the minimum viable test that would catch if these changes regressed.
Focus on the public interface that changed, not internals.
Write exactly ONE test file.

You MUST respond with valid YAML ONLY matching this schema:
test_file: path/to/test_file.ext
content: |
  full test code here
description: what this tests
"""

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        state = context.previous_output or {}

        task_data: dict[str, Any] = {
            "task_objective": context.objective,
            "implementation_summary": state.get("implementation_summary", "None"),
            "files_modified": state.get("files_modified", []),
            "files_created": state.get("files_created", []),
            "test_framework_command": context.extra.get("test_command", "unknown"),
        }

        file_context = ""
        if context.file_context:
            file_context = "\n\nModified Files Context:\n"
            for fname, content in context.file_context.items():
                file_context += f"\n--- {fname} ---\n{content}\n"

        user_content = self._yaml_block(task_data)
        if file_context:
            user_content += file_context
        user_content += "\n\nGenerate the regression test as YAML."

        return [self._system_msg(), self._user_msg(user_content)]

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Override run — YAML output does not need a special response_format."""
        return super().run(context, **kwargs)

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
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
            raw = yaml.safe_load(content)
            if not isinstance(raw, dict):
                raise ValueError("YAML root must be a mapping")
            validated = TestGenResponse(**raw)
            result = validated.model_dump()
        except (yaml.YAMLError, ValueError, ValidationError) as e:
            # Fallback: try JSON in case the model ignores format instructions
            try:
                raw = json.loads(content)
                validated = TestGenResponse(**raw)
                result = validated.model_dump()
            except (json.JSONDecodeError, ValidationError, TypeError):
                logger.warning(f"[SHIELD] Failed to parse test generation response: {e}")
                result = {
                    "test_file": "",
                    "content": "",
                    "description": f"Failed to generate: {e}",
                    "parse_error": True,
                }

        result["_agent"] = "testgen"
        result["_model"] = response.model
        result["_tokens"] = response.tokens_used
        result["_cost"] = response.cost

        return result
