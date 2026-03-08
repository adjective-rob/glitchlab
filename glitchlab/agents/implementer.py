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
            "description": "Use this to plan your approach. You must map out your search strategy and list the files you suspect you need to modify before taking action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_strategy": {
                        "type": "string", 
                        "description": "How you will find the code you need (e.g., 'I will grep for X to find all downstream imports')."
                    },
                    "execution_plan": {
                        "type": "string",
                        "description": "Step-by-step plan of which files to read and write."
                    }
                },
                "required": ["search_strategy", "execution_plan"]
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
            "name": "replace_in_file",
            "description": "Replace a specific exact string with a new string in an existing file. ALWAYS prefer this over write_file for existing files to avoid deleting code. The 'find' string MUST match the existing file content exactly.",
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
            "description": "Undo your changes to a file, restoring it to the version before you modified it. Use this when a write_file broke something and you need to start that file over.",
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
    },
    {
        "type": "function",
        "function": {
            "name": "find_references",
            "description": "Find all locations where a symbol is defined, called, or imported. More precise than search_grep — ignores matches in comments and strings. Use this for renaming, understanding callers, or checking impact of changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "The exact symbol name to find"},
                    "language": {"type": "string", "description": "Optional language filter (e.g. 'python', 'rust')"}
                },
                "required": ["symbol"]
            }
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
            "name": "ask_colleague",
            "description": "If you lack the expertise or need a specialized review (e.g., security, testing) before finishing, delegate a sub-task to a colleague.",
            "parameters": {
                "type": "object",
                "properties": {
                    "colleague": {
                        "type": "string", 
                        "enum": ["security", "debugger", "testgen", "archivist"],
                        "description": "The specialized agent to call"
                    },
                    "request": {
                        "type": "string",
                        "description": "Exactly what you need them to do or figure out for you."
                    }
                },
                "required": ["colleague", "request"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_function",
            "description": "Get the complete body of a function or method by name. Returns the full implementation including signature. Use this instead of read_file when you only need one function from a large file to save context window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "The function or method name"},
                    "file": {"type": "string", "description": "Optional file path to restrict the search"}
                },
                "required": ["symbol"]
            }
        }
    }               
]


class ImplementerAgent(BaseAgent):
    role = "implementer"

    system_prompt = """You are Patch, the surgical implementation engine.

You now operate in an agentic loop. You have tools to think, read, write, check, and rollback.
1. You MUST use the `think` tool to explain your step-by-step execution plan BEFORE you use the `write_file` tool for the first time.
2. DO NOT guess type signatures. If you need to know how a module works, use `read_file` or `get_function`.
3. For existing files, ALWAYS prefer `replace_in_file` to make surgical edits. Only use `write_file` if you are creating a brand new file or completely rewriting a very small one.
4. If you are unsure if your code is right, use `run_check` to run the compiler, linter, or tests.
5. When using write_file, you MUST output the ENTIRE file contents. NEVER use placeholders like 'rest of code here'. Doing so will delete the user's code.
6. If you make a mistake and break a file, use the `rollback_file` tool to undo your changes and start over.
7. Use `get_function` to read specific function bodies instead of `read_file` to save context space on large files.
8. Use `find_references` to understand where a symbol is defined or called before changing its signature.
9. When you are confident the plan is implemented, use the `done` tool.
"""

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        state = context.previous_output

        task_data: dict[str, Any] = {
            "task": context.objective,
            "plan": [
                {"step": s.get("step_number"), "description": s.get("description")}
                for s in state.get("plan_steps", [])
            ],
        }

        user_content = self._yaml_block(task_data)

        if context.file_context:
            user_content += "\n\nInitial file contents provided by router:\n"
            for fname, content in context.file_context.items():
                user_content += f"\n--- {fname} ---\n{content}\n"

        heuristics = context.extra.get("learned_heuristics")
        if heuristics:
            user_content += f"\n\n{heuristics}"

        user_content += "\n\nUse your tools to explore, implement, and verify this plan. When finished, call `done`."

        return [self._system_msg(), self._user_msg(user_content)]

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Override run to implement the Agentic Loop with Cognitive Monologue."""
        messages = self.build_messages(context)
        
        workspace_dir = Path(context.working_dir)
        tool_executor = context.extra.get("tool_executor")
        symbol_index = context.extra.get("symbol_index")
        
        modified_files = set()
        created_files = set()
        think_count = 0
        search_count = 0
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
                        
                        elif tname == "find_references" and len(content) > 500:
                            lines = content.splitlines()
                            messages[i]["content"] = "\n".join(lines[:30]) + "\n... [References compressed]"

                        elif tname == "get_function" and len(content) > 1000:
                            lines = content.splitlines()
                            head = "\n".join(lines[:20])
                            messages[i]["content"] = head + "\n... [Function body compressed]"

                        elif tname == "query_project_context" and len(content) > 500:
                            messages[i]["content"] = content[:500] + "\n... [Context compressed]"                                                  
                        
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

            # 3. Deterministic First Step: Force 'think' on step 0
            step_kwargs = dict(kwargs)
            if step == 0:
                step_kwargs["tool_choice"] = {"type": "function", "function": {"name": "think"}}

            response = self.router.complete(
                role=self.role,
                messages=messages,
                tools=IMPLEMENTER_TOOLS,
                **step_kwargs
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
                    res = "Strategy noted. Proceed with your plan."
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

                elif tc_name == "query_symbol_map":
                    query = tc_args.get("query", "").lower()
                    repo_index = context.extra.get("repo_index")
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
                        res = "Access Denied: You must use the `think` tool to explain your modifications before calling `write_file`."
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

                        if symbol_index:
                            symbol_index.invalidate(path)
                            
                        res = f"Successfully wrote {len(content)} characters to {path}."
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
                                
                                if symbol_index:
                                    symbol_index.invalidate(path)
                                    
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

                elif tc_name == "find_references":
                    symbol = tc_args.get("symbol")
                    language = tc_args.get("language")
                    if symbol_index:
                        refs = symbol_index.find_references(symbol, language)
                        if not refs:
                            res = f"No structural references found for '{symbol}'. Fall back to search_grep if needed."
                        else:
                            lines = [f"{r['file']}:{r['line']} [{r['kind']}] {r['context']}" for r in refs[:30]]
                            res = f"Found {len(refs)} references for '{symbol}':\n" + "\n".join(lines)
                            if len(refs) > 30:
                                res += f"\n... (truncated {len(refs)-30} more)"
                    else:
                        res = "AST parser unavailable. Please fall back to search_grep."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "get_function":
                    symbol = tc_args.get("symbol")
                    file_path = tc_args.get("file")
                    if symbol_index:
                        func_data = symbol_index.get_function_body(symbol, file_path)
                        if func_data:
                            res = f"Function '{symbol}' in {func_data['file']} (Lines {func_data['line_start']}-{func_data['line_end']}):\n\n{func_data['body']}"
                        else:
                            res = f"Function '{symbol}' not found. Check spelling or use search_grep."
                    else:
                        res = "AST parser unavailable. Please fall back to read_file."
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

                elif tc_name == "ask_colleague":
                    return {
                        "_status": "delegating",
                        "colleague": tc_args.get("colleague"),
                        "request": tc_args.get("request"),
                        "_messages": messages,
                        "tc_id": tc_id,
                        "tc_name": tc_name
                    }

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
                        "_messages": messages,
                    }

                else:
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": f"Error: Unknown tool '{tc_name}'."})

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