"""
📦 Semver Sam — Release + Version Guardian (v3.1 Tool-Loop)

Analyzes API surface area to decide on version bumps.
Surgically updates CHANGELOG.md.
Energy: accountant with neon sneakers.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


SAM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Analyze the implementation summary and diff to determine versioning impact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_impact": {
                        "type": "string",
                        "description": "Did any function signatures, variable names, or public exports change?"
                    },
                    "bump_logic": {
                        "type": "string",
                        "description": "Reasoning for major vs minor vs patch based on semver rules."
                    }
                },
                "required": ["api_impact", "bump_logic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file (like pyproject.toml or a modified module) to verify the current state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_grep",
            "description": "Search for usages of modified functions to see if they are used as public APIs elsewhere.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Surgically insert the changelog entry into CHANGELOG.md. Use this to avoid overwriting existing history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": "CHANGELOG.md"
                    },
                    "find": {
                        "type": "string",
                        "description": "The exact text to find (e.g., '# Changelog')"
                    },
                    "replace": {
                        "type": "string",
                        "description": "The new text including the changelog entry."
                    }
                },
                "required": ["path", "find", "replace"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_verdict",
            "description": "Finalize the release assessment and return structured data to the controller.",
            "parameters": {
                "type": "object",
                "properties": {
                    "version_bump": {
                        "type": "string",
                        "enum": ["none", "patch", "minor", "major"]
                    },
                    "reasoning": {"type": "string"},
                    "changelog_entry": {"type": "string"},
                    "breaking_changes": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "risk_summary": {"type": "string"}
                },
                "required": [
                    "version_bump",
                    "reasoning",
                    "changelog_entry",
                    "risk_summary"
                ]
            }
        }
    }
]


class ReleaseAgent(BaseAgent):
    role = "release"

    system_prompt = """You are Semver Sam, the release guardian. 
You operate in a tool-calling loop. 

1. Use `think` first to analyze if the changes are breaking (Major), additive (Minor), or internal (Patch).
2. Use `read_file` to check the actual code modified by the Implementer.
3. Use `replace_in_file` to update CHANGELOG.md surgically. 
4. Call `submit_verdict` when finished.

Semver Rules:
- major: Breaking changes to public API or signatures.
- minor: New features, non-breaking additions.
- patch: Internal refactors, bug fixes, no API change.
"""

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        state = context.previous_output or {}
        diff_text = context.extra.get("diff", "")

        diff_preview = (
            diff_text[:2000] + "\n... [Truncated]"
            if len(diff_text) > 2000
            else diff_text
        )

        user_content = f"""Analyze the version impact of these changes.

Task: {context.objective}
Mode: {state.get('mode', 'evolution')}
Implementation Summary: {state.get('implementation_summary', 'N/A')}

Diff Preview:

Investigate the modified files, update CHANGELOG.md surgically, and call `submit_verdict`."""

        return [self._system_msg(), self._user_msg(user_content)]

    def parse_response(
        self, response: RouterResponse, context: AgentContext
    ) -> dict[str, Any]:
        """Unused in v3.1 Tool-Loop."""
        return {}

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Execute the Semver Sam investigation loop."""
        messages = self.build_messages(context)
        workspace_dir = Path(context.working_dir)

        think_count = 0
        max_steps = 10

        for step in range(max_steps):
            logger.debug(f"[SEMVER] Loop Step {step+1}/{max_steps}...")

            # Context compression for long tool outputs
            for i in range(len(messages)):
                if messages[i].get("role") == "tool":
                    consumed = any(
                        m.get("role") == "assistant"
                        for m in messages[i + 1 :]
                    )
                    if consumed:
                        content = str(messages[i].get("content", ""))
                        if len(content) > 1000:
                            messages[i]["content"] = (
                                content[:500] + "\n... [Content compressed]"
                            )

            step_kwargs = dict(kwargs)
            if step == 0:
                step_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": "think"},
                }

            response = self.router.complete(
                role=self.role,
                messages=messages,
                tools=SAM_TOOLS,
                **step_kwargs,
            )

            assist_msg = {"role": "assistant"}
            if response.content:
                assist_msg["content"] = response.content
            if response.tool_calls:
                assist_msg["tool_calls"] = [
                    tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
                    for tc in response.tool_calls
                ]
            messages.append(assist_msg)

            if not response.tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "Analyze the changes or call `submit_verdict`.",
                    }
                )
                continue

            for tool_call in response.tool_calls:
                tc_id = tool_call.id
                tc_name = tool_call.function.name

                try:
                    tc_args = json.loads(
                        tool_call.function.arguments or "{}"
                    )
                except json.JSONDecodeError:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": tc_name,
                            "content": "Error: Invalid JSON.",
                        }
                    )
                    continue

                logger.info(f"[SEMVER] 🛠️ Tool call: {tc_name}")

                if tc_name == "think":
                    think_count += 1
                    res = "Impact analysis noted. Investigate files if needed."

                elif tc_name == "read_file":
                    path = tc_args.get("path")
                    try:
                        res = (workspace_dir / path).read_text(
                            encoding="utf-8"
                        )
                    except Exception as e:
                        res = f"Error reading file: {e}"

                elif tc_name == "search_grep":
                    pattern = tc_args.get("pattern")
                    try:
                        cmd = ["grep", "-rn", pattern, "."]
                        proc = subprocess.run(
                            cmd,
                            cwd=workspace_dir,
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        res = (
                            proc.stdout
                            if proc.stdout
                            else "No matches found."
                        )
                    except Exception as e:
                        res = f"Search failed: {e}"

                elif tc_name == "replace_in_file":
                    if think_count == 0:
                        res = "Access Denied: Use `think` first."
                    else:
                        path = tc_args.get("path", "CHANGELOG.md")
                        find_str = tc_args.get("find", "")
                        replace_str = tc_args.get("replace", "")
                        try:
                            fpath = workspace_dir / path
                            content = fpath.read_text()
                            if find_str in content:
                                fpath.write_text(
                                    content.replace(find_str, replace_str)
                                )
                                res = f"Successfully updated {path}."
                            else:
                                res = (
                                    f"Error: '{find_str}' not found in {path}."
                                )
                        except Exception as e:
                            res = f"Error: {e}"

                elif tc_name == "submit_verdict":
                    return {
                        "version_bump": tc_args.get("version_bump", "patch"),
                        "reasoning": tc_args.get("reasoning", ""),
                        "changelog_entry": tc_args.get("changelog_entry", ""),
                        "breaking_changes": tc_args.get("breaking_changes", []),
                        "risk_summary": tc_args.get("risk_summary", ""),
                        "_agent": "release",
                        "_model": response.model,
                        "_tokens": response.tokens_used,
                        "_cost": response.cost,
                    }

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": tc_name,
                        "content": str(res),
                    }
                )

        return {
            "version_bump": "patch",
            "reasoning": "Audit timeout.",
            "changelog_entry": "- Internal updates",
            "risk_summary": "Low (audit failed to conclude)",
        }