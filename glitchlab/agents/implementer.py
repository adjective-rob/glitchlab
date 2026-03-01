"""
🔧 Patch — The Implementer (v3.0 Tool-Loop Architecture)

Operates in a read-execute-observe loop. Now features a 'think' tool 
for multi-file coordination and complex reasoning.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


IMPLEMENTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Use this to plan your approach before taking action. Write out your reasoning about file dependencies, execution order, or potential issues. This helps you avoid errors in complex tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string", 
                        "description": "Your internal thoughts and step-by-step plan."
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
    },
    {
        "type": "function",
        "function": {
            "name": "search_grep",
            "description": "Search the codebase for a pattern. Returns matching lines with file paths and line numbers. Use this to find function definitions, imports, or usages before using read_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The text pattern to search for"},
                    "file_type": {"type": "string", "description": "Optional glob pattern, e.g., '*.py' or '*.rs'", "default": "*"}
                },
                "required": ["pattern"]
            }
        }
    }                 
]


class ImplementerAgent(BaseAgent):
    role = "implementer"

    system_prompt = """You are Patch, the surgical implementation engine.

You now operate in an agentic loop. You have tools to think, read, write, and check.
1. For complex tasks, use `think` first to map out dependencies.
2. DO NOT guess type signatures. If you need to know how a module works, use `read_file`.
3. Write one file at a time using `write_file`.
4. If you are unsure if your code is right, use `run_check` to run the compiler, linter, or tests.
5. When you are confident the plan is implemented, use the `done` tool.
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
        """Override run to implement the Agentic Loop with Cognitive Monologue."""
        messages = self.build_messages(context)
        
        workspace_dir = Path(context.working_dir)
        tool_executor = context.extra.get("tool_executor")
        
        modified_files = set()
        created_files = set()
        think_count = 0
        max_steps = 30
        
        for step in range(max_steps):
            logger.debug(f"[PATCH] Loop Step {step+1}/{max_steps}...")
            
            # 1. Proactive smart context compression
            for i in range(len(messages)):
                # Compress tool outputs after they've been consumed by the assistant
                if messages[i].get("role") == "tool":
                    consumed = any(m.get("role") == "assistant" for m in messages[i+1:])
                    if consumed:
                        content = str(messages[i].get("content", ""))
                        if "... [Content compressed" in content or "... [Search results compressed" in content:
                            continue  # Already compressed
                        
                        tname = messages[i].get("name")
                        
                        # Smart symbol extraction for read_file
                        if tname == "read_file" and len(content) > 1000:
                            lines = content.splitlines()
                            head = "\n".join(lines[:10])
                            tail = "\n".join(lines[-10:])
                            # Extract functions, classes, structs, etc.
                            symbols = [l.strip() for l in lines if l.strip().startswith(("def ", "class ", "async def ", "pub fn ", "struct ", "type ", "export "))]
                            sym_str = "\n".join(symbols[:20])
                            messages[i]["content"] = f"{head}\n\n... [Content compressed. Key symbols:]\n{sym_str}\n...\n{tail}"
                        
                        # Reference-only extraction for search_grep
                        elif tname == "search_grep" and len(content) > 500:
                            lines = content.splitlines()
                            refs = []
                            for l in lines:
                                parts = l.split(":")
                                if len(parts) >= 2:
                                    refs.append(f"{parts[0]}:{parts[1]}")
                            if refs:
                                messages[i]["content"] = "\n".join(refs[:30]) + "\n... [Search results compressed to references only]"
                            else:
                                messages[i]["content"] = content[:500] + "\n... [Search results compressed]"
                
                # Compress tool inputs (e.g. massive write_file contents) after consumption
                if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
                    consumed = any(m.get("role") == "tool" for m in messages[i+1:])
                    if consumed:
                        for tc in messages[i]["tool_calls"]:
                            if tc.get("function", {}).get("name") == "write_file":
                                try:
                                    args = json.loads(tc["function"]["arguments"])
                                    if "content" in args and len(str(args["content"])) > 200:
                                        lines_written = len(str(args["content"]).splitlines())
                                        path = args.get("path", "unknown")
                                        args["content"] = f"... [Content compressed: wrote {lines_written} lines to {path}]"
                                        tc["function"]["arguments"] = json.dumps(args)
                                except Exception:
                                    pass

            # 2. Rolling window search spiral guard
            # Look at the last 6 tool calls across all messages
            recent_tools = [m.get("name") for m in messages if m.get("role") == "tool"][-6:]
            search_count = recent_tools.count("search_grep")

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

                if tc_name == "think":
                    think_count += 1
                    if think_count > 3:
                        res = "Thinking limit reached. Please proceed with actions (read, write, or search)."
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
                    if search_count >= 3:
                        res = "You have searched multiple times recently. Consider using `think` to consolidate your findings or `read_file` to look closer."
                        messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})
                        continue

                    pattern = tc_args.get("pattern")
                    file_type = tc_args.get("file_type", "*")
                    try:
                        # Direct subprocess call for read-only universal search
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