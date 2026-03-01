"""
🐛 Reroute — The Debugger (v3.0 Tool-Loop Architecture)

Now operates in an agentic loop. Features a 'think' tool for root-cause
analysis, search_grep for exploration, and get_error for verification.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


DEBUGGER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Use this to analyze error logs, hypothesize root causes, or plan your fix before taking action. Write out your reasoning about the failure chain. This helps avoid incorrect patches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string", 
                        "description": "Your internal diagnosis and step-by-step fix plan."
                    }
                },
                "required": ["reasoning"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file to understand the failing logic or type signatures.",
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
            "name": "write_file",
            "description": "Write corrected content to a file to fix the bug.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_check",
            "description": "Run a shell command (e.g., a linter or compiler) to validate your fix.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_grep",
            "description": "Search the codebase for a pattern. Returns matching lines with file paths and line numbers. Use this to find where functions are called or where constants are defined.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The text pattern to search for"},
                    "file_type": {"type": "string", "description": "Optional glob pattern, e.g., '*.py' or '*.rs'", "default": "*"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_error",
            "description": "Re-run the failing test command and return fresh output. Use this to see if your partial fix changed the error.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that the bug is fixed and verified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diagnosis": {"type": "string", "description": "Short summary of what was failing"},
                    "root_cause": {"type": "string", "description": "The specific logic error found"},
                    "fix_summary": {"type": "string", "description": "What you changed to fix it"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
                },
                "required": ["diagnosis", "root_cause", "fix_summary", "confidence"]
            }
        }
    }
]

class DebuggerAgent(BaseAgent):
    role = "debugger"

    system_prompt = """You are Reroute, the surgical debug engine.

You now operate in an agentic loop. You have tools to think, investigate, fix, and verify.
1. Use `think` to hypothesize why a test is failing before you change code.
2. Use `get_error` to see the current failure.
3. Use `search_grep` if you don't know the exact file path.
4. Use `read_file` to examine the logic.
5. Use `write_file` to apply a surgical fix.
6. When the test passes, call `done`.

The test command you are debugging is: {test_command}
"""

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        state = context.previous_output or {}
        test_cmd = context.extra.get("test_command", "unknown")
        error_log = context.extra.get("error_output", "")[:1000]

        sys_prompt = self.system_prompt.format(test_command=test_cmd)

        user_content = f"""FAILURE DETECTED
Objective: {context.objective}
Failing Command: {test_cmd}

Initial Error Output:
{error_log}

Modified Files: {state.get('files_modified', [])}

Investigate and fix. Call `done` when the tests pass."""

        return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}]

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Execute the agentic debug loop with Cognitive Monologue."""
        messages = self.build_messages(context)
        workspace_dir = Path(context.working_dir)
        tool_executor = context.extra.get("tool_executor")
        test_cmd = context.extra.get("test_command")
        
        modified_files = set()
        created_files = set()
        think_count = 0
        max_steps = 10 

        for step in range(max_steps):
            logger.debug(f"[REROUTE] Loop Step {step+1}/{max_steps}...")
            
            response = self.router.complete(
                role=self.role,
                messages=messages,
                tools=DEBUGGER_TOOLS,
                **kwargs
            )

            # Append assistant message
            assist_msg = {"role": "assistant"}
            if response.content:
                assist_msg["content"] = response.content
            if response.tool_calls:
                assist_msg["tool_calls"] = [
                    tc.model_dump() if hasattr(tc, 'model_dump') else dict(tc) 
                    for tc in response.tool_calls
                ]
            messages.append(assist_msg)

            if not response.tool_calls:
                messages.append({"role": "user", "content": "Please use a tool to investigate or call `done`."})
                continue

            for tool_call in response.tool_calls:
                tc_id = tool_call.id
                tc_name = tool_call.function.name
                
                try:
                    tc_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": "Error: Invalid JSON."})
                    continue

                logger.info(f"[REROUTE] 🛠️ Tool call: {tc_name}")

                if tc_name == "think":
                    think_count += 1
                    if think_count > 3:
                        res = "Thinking limit reached. Please take action (read, write, or search)."
                    else:
                        res = "Reasoning noted. Continue when ready."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "read_file":
                    path = tc_args.get("path")
                    try:
                        content = (workspace_dir / path).read_text(encoding='utf-8')
                        res = f"Read {len(content)} characters from {path}:\n\n{content}"
                    except Exception as e:
                        res = f"Error reading file: {e}"
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "search_grep":
                    pattern = tc_args.get("pattern")
                    file_type = tc_args.get("file_type", "*")
                    try:
                        cmd = [
                            "grep", "-rn", 
                            f"--include={file_type}",
                            "--exclude-dir=.glitchlab", 
                            "--exclude-dir=__pycache__",
                            "--exclude-dir=node_modules",
                            "--exclude-dir=.git",
                            pattern, "."
                        ]
                        proc = subprocess.run(cmd, cwd=workspace_dir, capture_output=True, text=True, timeout=30)
                        lines = proc.stdout.splitlines()
                        if len(lines) > 50:
                            res = "\n".join(lines[:50]) + "\n(truncated, refine your search)"
                        else:
                            res = proc.stdout if proc.stdout else "No matches found."
                    except Exception as e:
                        res = f"Search failed: {e}"
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "write_file":
                    path = tc_args.get("path")
                    content = tc_args.get("content")
                    try:
                        fpath = workspace_dir / path
                        is_new = not fpath.exists()
                        fpath.parent.mkdir(parents=True, exist_ok=True)
                        fpath.write_text(content, encoding='utf-8')
                        
                        if is_new:
                            created_files.add(path)
                        else:
                            modified_files.add(path)
                        res = f"Successfully updated {path}."
                    except Exception as e:
                        res = f"Error writing file: {e}"
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "get_error" or (tc_name == "run_check" and not tc_args.get("command")):
                    if tool_executor:
                        tres = tool_executor.execute(test_cmd)
                        res = f"Exit {tres.returncode}\nSTDOUT: {tres.stdout}\nSTDERR: {tres.stderr}"
                    else:
                        res = "Error: No executor wired up."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "run_check":
                    cmd = tc_args.get("command")
                    if tool_executor:
                        tres = tool_executor.execute(cmd)
                        res = f"Exit {tres.returncode}\nSTDOUT: {tres.stdout}\nSTDERR: {tres.stderr}"
                    else:
                        res = "Error: Tool executor not wired up."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "done":
                    return {
                        "diagnosis": tc_args.get("diagnosis"),
                        "root_cause": tc_args.get("root_cause"),
                        "fix": {
                            "changes": [
                                {"file": f, "action": "modify", "_already_applied": True} for f in modified_files
                            ] + [
                                {"file": f, "action": "create", "_already_applied": True} for f in created_files
                            ]
                        },
                        "confidence": tc_args.get("confidence"),
                        "should_retry": True,
                        "summary": tc_args.get("fix_summary", "Done."),
                        "_agent": "debugger",
                        "_model": response.model,
                        "_tokens": response.tokens_used,
                        "_cost": response.cost,
                    }

        return {
            "diagnosis": "Max steps reached", 
            "root_cause": "JSON_TRUNCATION", 
            "should_retry": False, 
            "parse_error": True
        }

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        pass # Unused in tool-loop mode