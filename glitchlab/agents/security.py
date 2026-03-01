"""
🔒 Firewall Frankie — Security + Policy Guard (v3.1 Tool-Loop Architecture)

Scans for dangerous patterns.
Checks dependency diffs.
Investigates the codebase for hidden vulnerabilities.
Watches core boundaries.

Energy: cartoon cop with a magnifying glass.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


SECURITY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Use this to plan your security audit BEFORE taking action. Map out which modified files you need to read and what patterns you are hunting for.",
            "parameters": {
                "type": "object",
                "properties": {
                    "audit_plan": {
                        "type": "string",
                        "description": "Your step-by-step plan to verify the security of the recent changes."
                    },
                    "threat_model": {
                        "type": "string",
                        "description": "What specific security risks (e.g., injection, path traversal) are most likely introduced by this specific task?"
                    }
                },
                "required": ["audit_plan", "threat_model"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a modified file to check its context, imports, and data flow.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_grep",
            "description": "Search the codebase for a pattern. Useful for checking if a newly introduced variable shadows a global credential, or finding where a vulnerable function is called.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The text pattern to search for"
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Optional glob pattern, e.g., '*.py' or '*.rs'",
                        "default": "*"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_report",
            "description": "Signal that the security audit is complete and submit your final verdict.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["pass", "warn", "block"],
                        "description": "pass = clean. warn = should fix but safe to merge. block = MUST fix before PR."
                    },
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "severity": {
                                    "type": "string",
                                    "enum": ["critical", "high", "medium", "low", "info"]
                                },
                                "file": {"type": "string"},
                                "line": {"type": "integer"},
                                "description": {"type": "string"},
                                "recommendation": {"type": "string"}
                            },
                            "required": ["severity", "file", "description"]
                        }
                    },
                    "dependency_changes": {
                        "type": "object",
                        "properties": {
                            "added": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "removed": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "risk_assessment": {
                                "type": "string",
                                "enum": ["none", "low", "medium", "high"]
                            }
                        },
                        "required": ["added", "removed", "risk_assessment"]
                    },
                    "boundary_violations": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief security summary of your findings."
                    }
                },
                "required": [
                    "verdict",
                    "issues",
                    "dependency_changes",
                    "boundary_violations",
                    "summary"
                ]
            }
        }
    }
]


class SecurityAgent(BaseAgent):
    role = "security"

    system_prompt = """You are Firewall Frankie, the security guard inside GLITCHLAB.

You review code changes BEFORE they become a PR. You look for security issues, dangerous patterns, and policy violations.
You operate in a read-execute-observe loop. 

Rules:
1. Use the `think` tool first to build a threat model based on what files were changed.
2. Use `read_file` to inspect the FULL context of the modified files. Do not guess based on the diff snippet alone.
3. Use `search_grep` to trace data flow or check for cross-file vulnerabilities.
4. Be thorough but don't false-positive on idiomatic patterns. Severity must be honest.
5. When you have finished your audit, use the `submit_report` tool to output your JSON verdict.
"""

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        state = context.previous_output or {}
        diff_text = context.extra.get("diff", "")

        # Truncate large diffs to protect context window
        if len(diff_text) > 2000:
            diff_text = (
                diff_text[:2000]
                + "\n\n... [DIFF TRUNCATED. USE read_file TO SEE FULL CHANGES] ..."
            )

        files_modified = state.get("files_modified", [])
        files_created = state.get("files_created", [])
        impl_summary = state.get("implementation_summary", "No summary available")

        user_content = f"""Review these recent code changes for security and policy compliance.

Task: {context.objective}
Mode: {state.get('mode', 'evolution')}

Implementation summary: {impl_summary}
Files modified: {files_modified}
Files created: {files_created}
Protected paths: {context.extra.get('protected_paths', [])}

Diff Preview:

{diff_text}

Investigate the modified files using your tools. When satisfied, call `submit_report`."""

        return [self._system_msg(), self._user_msg(user_content)]

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Execute the agentic security loop."""
        messages = self.build_messages(context)
        workspace_dir = Path(context.working_dir)

        think_count = 0
        max_steps = 15

        for step in range(max_steps):
            logger.debug(f"[FRANKIE] Loop Step {step+1}/{max_steps}...")

            # Context compression for large tool outputs
            for i in range(len(messages)):
                if messages[i].get("role") == "tool":
                    consumed = any(
                        m.get("role") == "assistant"
                        for m in messages[i + 1 :]
                    )
                    if consumed:
                        content = str(messages[i].get("content", ""))
                        if (
                            len(content) > 1000
                            and "... [Content compressed" not in content
                        ):
                            messages[i]["content"] = (
                                content[:500]
                                + "\n... [Content compressed for context window]"
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
                tools=SECURITY_TOOLS,
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
                        "content": "Please investigate using tools or call `submit_report`.",
                    }
                )
                continue

            for tool_call in response.tool_calls:
                tc_id = tool_call.id
                tc_name = tool_call.function.name

                try:
                    tc_args = json.loads(tool_call.function.arguments or "{}")
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

                logger.info(f"[FRANKIE] 🛠️ Tool call: {tc_name}")

                if tc_name == "think":
                    think_count += 1
                    res = "Threat model noted. Proceed with your file investigation."
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": tc_name,
                            "content": res,
                        }
                    )

                elif tc_name == "read_file":
                    path = tc_args.get("path")
                    try:
                        content = (workspace_dir / path).read_text(
                            encoding="utf-8"
                        )
                        res = f"Read {len(content)} chars from {path}:\n\n{content}"
                    except Exception as e:
                        res = f"Error reading file: {e}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": tc_name,
                            "content": res,
                        }
                    )

                elif tc_name == "search_grep":
                    pattern = tc_args.get("pattern")
                    file_type = tc_args.get("file_type", "*")
                    try:
                        cmd = [
                            "grep",
                            "-rn",
                            f"--include={file_type}",
                            "--exclude-dir=.glitchlab",
                            "--exclude-dir=__pycache__",
                            "--exclude-dir=.git",
                            pattern,
                            ".",
                        ]
                        proc = subprocess.run(
                            cmd,
                            cwd=workspace_dir,
                            capture_output=True,
                            text=True,
                            timeout=15,
                        )
                        res = (
                            proc.stdout
                            if proc.stdout
                            else "No matches found."
                        )
                        if len(res.splitlines()) > 50:
                            res = (
                                "\n".join(res.splitlines()[:50])
                                + "\n... (truncated)"
                            )
                    except Exception as e:
                        res = f"Search failed: {e}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": tc_name,
                            "content": res,
                        }
                    )

                elif tc_name == "submit_report":
                    return {
                        "verdict": tc_args.get("verdict", "warn"),
                        "issues": tc_args.get("issues", []),
                        "dependency_changes": tc_args.get(
                            "dependency_changes",
                            {"added": [], "removed": [], "risk_assessment": "none"},
                        ),
                        "boundary_violations": tc_args.get(
                            "boundary_violations", []
                        ),
                        "summary": tc_args.get("summary", "Done."),
                        "_agent": "security",
                        "_model": response.model,
                        "_tokens": response.tokens_used,
                        "_cost": response.cost,
                    }

        return {
            "verdict": "warn",
            "issues": [
                {
                    "severity": "info",
                    "file": "system",
                    "description": "Security agent ran out of steps.",
                    "recommendation": "Review manually.",
                }
            ],
            "dependency_changes": {
                "added": [],
                "removed": [],
                "risk_assessment": "none",
            },
            "boundary_violations": [],
            "summary": "Security review timed out.",
            "parse_error": True,
        }

    def parse_response(
        self, response: RouterResponse, context: AgentContext
    ) -> dict[str, Any]:
        pass  # Unused because we overrode run()