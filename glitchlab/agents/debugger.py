"""
🐛 Reroute — The Debugger (v3.0 Tool-Loop Architecture)

Now operates in an agentic loop. Can read failing code, run tests to see 
evolving errors, and apply surgical fixes directly via tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse

DEBUGGER_TOOLS = [
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

You are invoked when tests fail. You have tools to investigate, fix, and verify.
1. Use `get_error` to see the current failure.
2. Use `read_file` to examine the code around the failure.
3. Use `write_file` to apply a surgical fix.
4. Use `get_error` or `run_check` again to verify the fix.
5. When the test passes, call `done`.

The test command you are debugging is: {test_command}
You can call `get_error` with no arguments to re-run this specific command.
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

Use your tools to investigate and fix the bug. Call `done` when the tests pass."""

        return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}]

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        messages = self.build_messages(context)
        workspace_dir = Path(context.working_dir)
        tool_executor = context.extra.get("tool_executor")
        test_cmd = context.extra.get("test_command")
        
        modified_files = set()
        max_steps = 10  # Surgical limit

        for step in range(max_steps):
            response = self.router.complete(
                role=self.role,
                messages=messages,
                tools=DEBUGGER_TOOLS,
                **kwargs
            )

            assist_msg = {"role": "assistant"}
            if response.content: assist_msg["content"] = response.content
            if response.tool_calls:
                assist_msg["tool_calls"] = [
                    tc.model_dump() if hasattr(tc, 'model_dump') else dict(tc) 
                    for tc in response.tool_calls
                ]
            messages.append(assist_msg)

            if not response.tool_calls:
                messages.append({"role": "user", "content": "Please use a tool or call `done`."})
                continue

            for tool_call in response.tool_calls:
                tc_id = tool_call.id
                tc_name = tool_call.function.name
                tc_args = json.loads(tool_call.function.arguments or "{}")

                logger.info(f"[REROUTE] 🛠️ Tool: {tc_name}")

                if tc_name == "read_file":
                    path = tc_args.get("path")
                    try:
                        content = (workspace_dir / path).read_text(encoding='utf-8')
                        res = f"Content of {path}:\n{content}"
                    except Exception as e: res = f"Error: {e}"
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "write_file":
                    path, content = tc_args.get("path"), tc_args.get("content")
                    try:
                        fpath = workspace_dir / path
                        fpath.parent.mkdir(parents=True, exist_ok=True)
                        fpath.write_text(content, encoding='utf-8')
                        modified_files.add(path)
                        res = f"Successfully updated {path}."
                    except Exception as e: res = f"Error: {e}"
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "get_error" or (tc_name == "run_check" and not tc_args.get("command")):
                    if tool_executor:
                        tres = tool_executor.execute(test_cmd)
                        res = f"Exit {tres.returncode}\nSTDOUT: {tres.stdout}\nSTDERR: {tres.stderr}"
                    else: res = "Error: No executor."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "run_check":
                    cmd = tc_args.get("command")
                    if tool_executor:
                        tres = tool_executor.execute(cmd)
                        res = f"Exit {tres.returncode}\nSTDOUT: {tres.stdout}\nSTDERR: {tres.stderr}"
                    else: res = "Error: No executor."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "done":
                    return {
                        "diagnosis": tc_args.get("diagnosis"),
                        "root_cause": tc_args.get("root_cause"),
                        "fix": {
                            "changes": [{"file": f, "action": "modify", "_already_applied": True} for f in modified_files]
                        },
                        "confidence": tc_args.get("confidence"),
                        "should_retry": True,
                        "summary": tc_args.get("fix_summary"),
                        "_agent": "debugger",
                        "_model": response.model,
                        "_tokens": response.tokens_used
                    }

        return {"diagnosis": "Max steps reached", "root_cause": "JSON_TRUNCATION", "should_retry": False, "parse_error": True}

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        pass