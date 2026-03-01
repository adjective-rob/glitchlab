"""
📚 Archivist Nova — Docs + ADR Writer (v3.1 Tool-Loop Architecture)

Captures design decisions after successful PRs.
Updates architecture notes.
Keeps future-you sane.

Energy: library robot with LED eyes.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


NOVA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Plan your documentation strategy BEFORE you change any files. Decide what docs need updates and whether an ADR is warranted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_plan": {
                        "type": "string",
                        "description": "Step-by-step plan of which docs you will read/update and why."
                    },
                    "adr_decision": {
                        "type": "string",
                        "description": "Should we write an ADR? If yes, what is the ADR about and what decision is being recorded?"
                    }
                },
                "required": ["doc_plan", "adr_decision"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read an existing documentation file to understand current content before editing.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_grep",
            "description": "Search the repo for references (e.g., architecture notes, module names, ADR index links) to update documentation consistently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The text pattern to search for"},
                    "file_type": {"type": "string", "description": "Optional glob pattern, e.g., '*.md'", "default": "*"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new documentation file (or overwrite a small doc file). Prefer replace_in_file for existing large docs.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Make a surgical edit by replacing an exact string with new content. Prefer this for existing docs to avoid deleting history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "find": {"type": "string", "description": "Exact existing text to find (must match whitespace)."},
                    "replace": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "find", "replace"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Finalize documentation work and submit structured notes to the controller.",
            "parameters": {
                "type": "object",
                "properties": {
                    "architecture_notes": {"type": "string", "description": "High-level architecture notes for future maintainers."},
                    "should_write_adr": {"type": "boolean", "description": "Whether an ADR was needed/written."},
                    "adr": {
                        "description": "ADR data (string markdown or structured object). Use null if none.",
                        "anyOf": [{"type": "string"}, {"type": "object"}, {"type": "null"}],
                    },
                    "doc_updates": {
                        "type": "array",
                        "description": "List of documentation files updated/created.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "action": {"type": "string", "enum": ["create", "modify"]},
                                "summary": {"type": "string"},
                            },
                            "required": ["path", "action"],
                        },
                    },
                },
                "required": ["architecture_notes", "should_write_adr", "doc_updates"],
            },
        },
    },
]


class ArchivistAgent(BaseAgent):
    role = "archivist"

    system_prompt = """You are Archivist Nova, the documentation engine inside GLITCHLAB.

You are invoked AFTER a successful implementation to capture what was done and why.
You operate EXCLUSIVELY in a tool-calling loop.

Rules:
1. MANDATORY START: You MUST use the `think` tool first to plan your documentation strategy.
2. SURGICAL UPDATES: If updating an existing file (like README.md), you MUST use `read_file` first, then use `replace_in_file` for surgical updates.
3. NO TRUNCATION: Never use `write_file` on large existing documents. Prefer `replace_in_file` so you don't delete history.
4. ADR POLICY: Write ADRs for any change that affects architecture, public API, or introduces new patterns.
5. FINALIZATION: When all files are updated via tools, call the `done` tool to submit your final architectural notes and ADR data.
"""

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        # v2: previous_output is TaskState.to_agent_summary("archivist")
        # Contains: task_id, objective, mode, risk_level,
        #           plan_steps, files_modified, implementation_summary, version_bump
        state = context.previous_output or {}

        files_modified = state.get("files_modified", [])
        files_text = "\n".join(f"- {f}" for f in files_modified) if files_modified else "- None"

        plan_steps = state.get("plan_steps", [])
        steps_text = ""
        for step in plan_steps:
            steps_text += (
                f"\n- Step {step.get('step_number', '?')}: "
                f"{step.get('description', 'no description')}"
            )

        existing_docs = context.extra.get("existing_docs", [])
        existing_docs_text = "\n".join(f"- {f}" for f in existing_docs) if existing_docs else "- (none provided)"

        user_content = f"""A task has been successfully completed. Document it.

Task: {context.objective}
Task ID: {context.task_id}
Mode: {state.get('mode', 'evolution')}
Risk level: {state.get('risk_level', 'unknown')}
Version bump: {state.get('version_bump', 'unknown')}

Implementation summary:
{state.get('implementation_summary', 'No summary')}

Plan steps:
{steps_text}

Files modified:
{files_text}

Existing docs in repo:
{existing_docs_text}

Use your tools to update/create documentation and ADRs as needed.
When finished, call `done` with architecture_notes, should_write_adr, adr (or null), and doc_updates."""
        return [self._system_msg(), self._user_msg(user_content)]

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Execute the Archivist Nova documentation loop."""
        messages = self.build_messages(context)
        workspace_dir = Path(context.working_dir)

        think_count = 0
        max_steps = 20

        modified_files: set[str] = set()
        created_files: set[str] = set()

        for step in range(max_steps):
            logger.debug(f"[NOVA] Loop Step {step+1}/{max_steps}...")

            # Context compression for long tool outputs
            for i in range(len(messages)):
                if messages[i].get("role") == "tool":
                    consumed = any(m.get("role") == "assistant" for m in messages[i + 1 :])
                    if consumed:
                        content = str(messages[i].get("content", ""))
                        if len(content) > 1200 and "... [Content compressed" not in content:
                            messages[i]["content"] = content[:600] + "\n... [Content compressed for context window]"

            # Force 'think' on step 0
            step_kwargs = dict(kwargs)
            if step == 0:
                step_kwargs["tool_choice"] = {"type": "function", "function": {"name": "think"}}

            response = self.router.complete(
                role=self.role,
                messages=messages,
                tools=NOVA_TOOLS,
                **step_kwargs,
            )

            assist_msg: dict[str, Any] = {"role": "assistant"}
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
                    {"role": "user", "content": "Please use your tools to update docs/ADRs, or call `done` when finished."}
                )
                continue

            for tool_call in response.tool_calls:
                tc_id = tool_call.id
                tc_name = tool_call.function.name

                try:
                    tc_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": "Error: Invalid JSON in arguments."}
                    )
                    continue

                logger.info(f"[NOVA] 🛠️ Tool call: {tc_name}")

                if tc_name == "think":
                    think_count += 1
                    res = "Plan acknowledged. Proceed with documentation updates."

                elif tc_name == "read_file":
                    path = tc_args.get("path")
                    try:
                        content = (workspace_dir / path).read_text(encoding="utf-8")
                        res = f"Read {len(content)} chars from {path}:\n\n{content}"
                    except Exception as e:
                        res = f"Error reading file: {e}"

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
                            "--exclude-dir=node_modules",
                            "--exclude-dir=.git",
                            pattern,
                            ".",
                        ]
                        proc = subprocess.run(cmd, cwd=workspace_dir, capture_output=True, text=True, timeout=20)
                        out = proc.stdout if proc.stdout else "No matches found."
                        lines = out.splitlines()
                        if len(lines) > 60:
                            out = "\n".join(lines[:60]) + "\n... (truncated)"
                        res = out
                    except Exception as e:
                        res = f"Search failed: {e}"

                elif tc_name == "write_file":
                    if think_count == 0:
                        res = "Access Denied: Use `think` first before writing documentation."
                    else:
                        path = tc_args.get("path")
                        content = tc_args.get("content", "")
                        try:
                            fpath = workspace_dir / path
                            is_new = not fpath.exists()
                            fpath.parent.mkdir(parents=True, exist_ok=True)
                            fpath.write_text(content, encoding="utf-8")
                            if is_new:
                                created_files.add(path)
                            else:
                                modified_files.add(path)
                            res = f"Successfully wrote {len(content)} chars to {path}."
                        except Exception as e:
                            res = f"Error writing file: {e}"

                elif tc_name == "replace_in_file":
                    if think_count == 0:
                        res = "Access Denied: Use `think` first before modifying documentation."
                    else:
                        path = tc_args.get("path")
                        find_str = tc_args.get("find", "")
                        replace_str = tc_args.get("replace", "")
                        try:
                            fpath = workspace_dir / path
                            if not fpath.exists():
                                res = f"Error: {path} does not exist."
                            else:
                                content = fpath.read_text(encoding="utf-8")
                                if find_str not in content:
                                    res = (
                                        "Error: The exact 'find' string was not found. "
                                        "Use read_file to copy the exact text including whitespace."
                                    )
                                else:
                                    count = content.count(find_str)
                                    fpath.write_text(content.replace(find_str, replace_str), encoding="utf-8")
                                    modified_files.add(path)
                                    res = f"Success: Replaced {count} occurrence(s) in {path}."
                        except Exception as e:
                            res = f"Error replacing in file: {e}"

                elif tc_name == "done":
                    # Return structured output expected by controller
                    return {
                        "adr": tc_args.get("adr", None),
                        "doc_updates": tc_args.get("doc_updates", []),
                        "should_write_adr": tc_args.get("should_write_adr", False),
                        "architecture_notes": tc_args.get("architecture_notes", ""),
                        "_agent": "archivist",
                        "_model": response.model,
                        "_tokens": response.tokens_used,
                        "_cost": response.cost,
                        # Optional helpful extras without breaking existing deps:
                        "_files_touched": sorted(modified_files | created_files),
                    }

                else:
                    res = f"Error: Unknown tool '{tc_name}'."

                messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": str(res)})

        logger.warning("[NOVA] Loop exhausted without calling `done`.")
        return {
            "adr": None,
            "doc_updates": [],
            "should_write_adr": False,
            "architecture_notes": "Archivist hit max step limit without finalizing.",
            "parse_error": True,
            "_agent": "archivist",
        }

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        pass  # Unused because we overrode run()