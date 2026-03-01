"""
🔧 Patch — The Implementer (v3.0 Tool-Loop Architecture)

Operates in a read-execute-observe loop. Can pull context on demand,
write files individually, and run syntax checks before committing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


IMPLEMENTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the workspace to get type signatures or context.",
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
            "description": "Write complete content to a file in the workspace (creates or overwrites).",
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
            "description": "Run a shell command (like a linter, compiler, or tests) to validate your changes. e.g., 'cargo check' or 'python -m pytest'",
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
            "name": "done",
            "description": "Signal that implementation is complete and you have fulfilled the plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "commit_message": {"type": "string", "description": "Conventional commit message"},
                    "summary": {"type": "string", "description": "What you accomplished"}
                },
                "required": ["commit_message", "summary"]
            }
        }
    }
]

class ImplementerAgent(BaseAgent):
    role = "implementer"

    system_prompt = """You are Patch, the surgical implementation engine.

You now operate in an agentic loop. You have tools to read files, write files, and run checks.
1. DO NOT guess type signatures. If you need to know how a module works, use `read_file`.
2. Write one file at a time using `write_file`.
3. If you are unsure if your code is right, use `run_check` to run the compiler, linter, or tests.
4. When you are confident the plan is implemented, use the `done` tool.
"""

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        state = context.previous_output
        
        steps_text = ""
        for step in state.get("plan_steps", []):
            steps_text += f"\nStep {step.get('step_number')}: {step.get('description')}\n"

        file_context = ""
        if context.file_context:
            file_context = "\n\nInitial file contents provided by router:\n"
            for fname, content in context.file_context.items():
                file_context += f"\n--- {fname} ---\n{content}\n"

        user_content = f"""Task: {context.objective}
Plan: {steps_text}
{file_context}

Use your tools to explore, implement, and verify this plan. When finished, call `done`."""

        return [self._system_msg(), self._user_msg(user_content)]

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Override run to implement the Agentic Loop instead of a single shot."""
        messages = self.build_messages(context)
        
        workspace_dir = Path(context.working_dir)
        tool_executor = context.extra.get("tool_executor")
        
        modified_files = set()
        created_files = set()
        
        max_steps = 15
        
        for step in range(max_steps):
            logger.debug(f"[PATCH] Loop Step {step+1}/{max_steps}...")
            
            response = self.router.complete(
                role=self.role,
                messages=messages,
                tools=IMPLEMENTER_TOOLS,
                **kwargs
            )

            # Append assistant message
            assist_msg = {"role": "assistant"}
            if response.content:
                assist_msg["content"] = response.content
            if response.tool_calls:
                # Format litellm tool_calls to dict for the chat history
                assist_msg["tool_calls"] = [
                    tc.model_dump() if hasattr(tc, 'model_dump') else dict(tc) 
                    for tc in response.tool_calls
                ]
            messages.append(assist_msg)

            if not response.tool_calls:
                # If the LLM just talks without using tools, nudge it.
                messages.append({"role": "user", "content": "Please use your tools to take action, or call `done` if you are finished."})
                continue

            for tool_call in response.tool_calls:
                tc_id = tool_call.id
                tc_name = tool_call.function.name
                
                try:
                    tc_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": "Error: Invalid JSON in arguments."})
                    continue

                logger.info(f"[PATCH] 🛠️ Tool call: {tc_name}")

                if tc_name == "read_file":
                    path = tc_args.get("path")
                    try:
                        content = (workspace_dir / path).read_text(encoding='utf-8')
                        res = f"Read {len(content)} characters:\n\n{content}"
                    except Exception as e:
                        res = f"Error reading file: {e}"
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
                            
                        res = f"Successfully wrote {len(content)} characters to {path}."
                    except Exception as e:
                        res = f"Error writing file: {e}"
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "run_check":
                    cmd = tc_args.get("command")
                    if tool_executor:
                        try:
                            # Use the sandboxed executor
                            tool_res = tool_executor.execute(cmd)
                            res = f"Exit code: {tool_res.returncode}\nSTDOUT:\n{tool_res.stdout}\nSTDERR:\n{tool_res.stderr}"
                        except Exception as e:
                            res = f"Execution blocked or failed: {e}"
                    else:
                        res = "Error: Tool executor not wired up."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "done":
                    # Exit the loop!
                    return {
                        "changes": [
                            {"file": f, "action": "modify", "_already_applied": True} for f in modified_files
                        ] + [
                            {"file": f, "action": "create", "_already_applied": True} for f in created_files
                        ],
                        "tests_added": [],
                        "commit_message": tc_args.get("commit_message", "chore: automated implementation"),
                        "summary": tc_args.get("summary", "Done."),
                        "_agent": "implementer",
                        "_model": response.model,
                        "_tokens": response.tokens_used,
                        "_cost": response.cost,
                    }

        # If it hits max steps without calling 'done'
        logger.warning("[PATCH] Loop exhausted without calling `done`.")
        return {
            "changes": [],
            "tests_added": [],
            "commit_message": "chore: partial implementation",
            "summary": "Implementer hit max step limit.",
            "parse_error": True,
        }

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        pass # Unused because we overrode run()