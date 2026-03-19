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

from glitchlab.event_bus import bus

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.context_compressor import compress_stale_messages, hard_compact_messages
from glitchlab.router import RouterResponse


DEBUGGER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Use this to analyze the error output and plan your investigation BEFORE taking action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string", 
                        "description": "Based on the error log, what do you think is actually broken?"
                    },
                    "investigation_plan": {
                        "type": "string",
                        "description": "Step-by-step plan of which files to read or search to verify your hypothesis."
                    }
                },
                "required": ["hypothesis", "investigation_plan"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents. Use start_line/end_line to read a specific range of a large file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "description": "First line to return (1-indexed, inclusive). Optional."},
                    "end_line": {"type": "integer", "description": "Last line to return (1-indexed, inclusive). Optional."}
                },
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
            "name": "replace_in_file",
            "description": "Replace a specific exact string with a new string in an existing file. ALWAYS prefer this over write_file to avoid accidentally deleting code. The 'find' string MUST match the existing file content exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "find": {"type": "string", "description": "The exact existing text to find. Must match whitespace and indentation perfectly."},
                    "replace": {"type": "string", "description": "The new text to replace it with."}
                },
                "required": ["path", "find", "replace"]
            }
        }
    },   
    {
        "type": "function",
        "function": {
            "name": "rollback_file",
            "description": "Undo your changes to a file, restoring it to the version before you modified it. Use this when your fix broke something worse and you need to start that file over.",
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
            "name": "query_symbol_map",
            "description": "Query the repository's structural map to find where specific classes, functions, or modules are located. Use this to avoid blind grepping.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The name of the symbol or file to look for (e.g., 'Workspace', 'controller.py')"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_project_context",
            "description": "Query the project's architectural decisions, constraints, and stack info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "e.g., 'authentication', 'database'"},
                    "scope": {"type": "string", "description": "Directory path to filter rules by"}
                }
            }
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

_READ_FILE_TRUNCATE_THRESHOLD = 300
_READ_FILE_HEAD_LINES = 30
_READ_FILE_TAIL_LINES = 30


class DebuggerAgent(BaseAgent):
    role = "debugger"

    system_prompt = """You are Reroute, the surgical debug engine.

You now operate in an agentic loop. You have tools to think, investigate, fix, and verify.
1. You MUST use the `think` tool to state your hypothesis and investigation plan BEFORE taking any actions.
2. Use `get_error` to see the current failure.
3. Use `search_grep` if you don't know the exact file path.
4. Use `read_file` to examine the logic.
5. ALWAYS prefer `replace_in_file` to apply surgical fixes. Only use `write_file` if you are completely rewriting a file.
6. If you make a mistake and break a file further, use the `rollback_file` tool to undo your changes.
7. When the test passes, call `done`.

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
        search_count = 0
        fast_mode = context.extra.get("fast_mode", False)
        max_steps = 8 if fast_mode else 15

        for step in range(max_steps):
            logger.debug(f"[REROUTE] Loop Step {step+1}/{max_steps}...")
            
            # 1. Proactive smart context compression
            compress_stale_messages(messages)
            hard_compact_messages(messages)

            # 2. Rolling window search spiral guard
            # Look at the last 6 tool calls across all messages
            recent_tools = [m.get("name") for m in messages if m.get("role") == "tool"][-6:]
            search_count = recent_tools.count("search_grep")

            # 3. Deterministic First Step: Force 'think' on step 0
            step_kwargs = dict(kwargs)
            if step == 0:
                step_kwargs["tool_choice"] = {"type": "function", "function": {"name": "think"}}

            response = self.router.complete(
                role=self.role,
                messages=messages,
                tools=DEBUGGER_TOOLS,
                **step_kwargs
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
                bus.emit(
                    event_type="agent.tool_called",
                    payload={
                        "tool_name": tc_name,
                        "tool_args_keys": list(tc_args.keys()),
                    },
                    agent_id=self.role,
                )

                if tc_name == "think":
                    think_count += 1
                    res = "Hypothesis noted. Proceed with your investigation plan."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "read_file":
                    path = tc_args.get("path")
                    start_line = tc_args.get("start_line")
                    end_line = tc_args.get("end_line")
                    try:
                        content = (workspace_dir / path).read_text(encoding='utf-8')
                        file_lines = content.splitlines()
                        line_count = len(file_lines)

                        if start_line or end_line:
                            s = max(0, (start_line or 1) - 1)
                            e = min(line_count, end_line or line_count)
                            numbered = [f"{i}: {line}" for i, line in enumerate(file_lines[s:e], start=s + 1)]
                            res = f"Read {path} lines {s + 1}-{e} of {line_count}:\n\n" + "\n".join(numbered)
                        elif line_count > _READ_FILE_TRUNCATE_THRESHOLD:
                            head = "\n".join(file_lines[:_READ_FILE_HEAD_LINES])
                            tail = "\n".join(file_lines[-_READ_FILE_TAIL_LINES:])
                            res = (
                                f"Read {path} ({line_count} lines, truncated to first "
                                f"{_READ_FILE_HEAD_LINES} + last {_READ_FILE_TAIL_LINES}):\n\n"
                                f"{head}\n\n"
                                f"... ({line_count - _READ_FILE_HEAD_LINES - _READ_FILE_TAIL_LINES} lines omitted. "
                                f"Use read_file with start_line/end_line to see specific sections.) ...\n\n"
                                f"{tail}"
                            )
                        else:
                            res = f"Read {len(content)} characters from {path}:\n\n{content}"
                    except Exception as e:
                        res = f"Error reading file: {e}"
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "query_project_context":
                    topic = tc_args.get("topic", "")
                    scope = tc_args.get("scope", "")
                    prelude = context.extra.get("prelude")
                    
                    if prelude:
                        res = prelude.query(topic=topic, scope=scope)
                    else:
                        res = "Error: Prelude context not wired up."
                        
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "search_grep":
                    if search_count >= 3:
                        res = "You have searched multiple times recently. Consider using `think` to consolidate your findings or `read_file` to look closer."
                        messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})
                        continue

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

                elif tc_name == "query_symbol_map":
                    query = tc_args.get("query", "").lower()
                    repo_index = context.extra.get("repo_index") # <--- FIX IS HERE
                    if repo_index:
                        results = []
                        for path, entry in repo_index.files.items():
                            if query in path.lower() or any(query in s.lower() for s in entry.symbols):
                                symbol_match = [s for s in entry.symbols if query in s.lower()]
                                results.append(f"- {path} (Symbols: {', '.join(symbol_match[:5])})")
                        
                        res = "\n".join(results[:20]) if results else "No matches found in the structural map."
                    else:
                        res = "Structural map (RepoIndex) is unavailable."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "write_file":
                    if think_count == 0:
                        res = "Access Denied: You must use the `think` tool to state your hypothesis before modifying code."
                        messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})
                        continue
                        
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

                elif tc_name == "replace_in_file":
                    if think_count == 0:
                        res = "Access Denied: You must use the `think` tool to state your hypothesis before modifying code."
                        messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})
                        continue
                        
                    path = tc_args.get("path")
                    find_str = tc_args.get("find", "")
                    replace_str = tc_args.get("replace", "")
                    
                    try:
                        fpath = workspace_dir / path
                        if not fpath.exists():
                            res = f"Error: {path} does not exist."
                        else:
                            content = fpath.read_text(encoding='utf-8')
                            if find_str not in content:
                                res = "Error: The exact 'find' string was not found. You must match spaces and indentation exactly. Use read_file to check the exact text."
                            else:
                                count = content.count(find_str)
                                new_content = content.replace(find_str, replace_str)
                                fpath.write_text(new_content, encoding='utf-8')
                                modified_files.add(path)
                                res = f"Success: Replaced {count} occurrence(s) in {path}."
                    except Exception as e:
                        res = f"Error replacing in file: {e}"
                        
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})
                    
                elif tc_name == "rollback_file":
                    path = tc_args.get("path")
                    try:
                        fpath = workspace_dir / path
                        if path in created_files:
                            if fpath.exists():
                                fpath.unlink()
                            created_files.discard(path)
                            res = f"Rolled back {path} (deleted newly created file)."
                        elif path in modified_files:
                            subprocess.run(
                                ["git", "checkout", "--", path],
                                cwd=workspace_dir,
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            modified_files.discard(path)
                            res = f"Rolled back {path} to original version."
                        else:
                            res = f"Error: {path} has not been modified or created by you."
                    except Exception as e:
                        res = f"Error rolling back file: {e}"
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "get_error" or (tc_name == "run_check" and not tc_args.get("command")):
                    if tool_executor:
                        try:
                            # Use the sandboxed executor, passing IDs for Zephyr attestation
                            tres = tool_executor.execute(
                                command=test_cmd,
                                run_id=context.run_id,
                                agent_id=self.role
                            )
                            res = f"Exit {tres.returncode}\nSTDOUT: {tres.stdout}\nSTDERR: {tres.stderr}"
                            if tres.returncode != 0:
                                res += "\n\nTip: use `rollback_file` if you need to undo a broken change."
                        except Exception as e:
                            res = f"Execution blocked or failed: {e}"
                    else:
                        res = "Error: No executor wired up."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "run_check":
                    cmd = tc_args.get("command")
                    if tool_executor:
                        try:
                            # Use the sandboxed executor, passing IDs for Zephyr attestation
                            tool_res = tool_executor.execute(
                                command=cmd,
                                run_id=context.run_id,
                                agent_id=self.role
                            )
                            res = f"Exit code: {tool_res.returncode}\nSTDOUT:\n{tool_res.stdout}\nSTDERR:\n{tool_res.stderr}"
                            if tool_res.returncode != 0:
                                res += "\n\nTip: use `rollback_file` if you need to undo a broken change."
                        except Exception as e:
                            res = f"Execution blocked or failed: {e}"
                    else:
                        res = "Error: Tool executor not wired up."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "done":
                    bus.emit(
                        event_type="agent.done",
                        payload={
                            "tool_name": "done",
                            "loop_steps": step + 1,
                        },
                        agent_id=self.role,
                    )
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