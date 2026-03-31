"""
GLITCHLAB Auditor — Task Writer

Takes structured findings from the scanner and generates
well-scoped GLITCHLAB task YAML files using an Agentic Loop.

Now features 'think', 'read_file', 'search_grep', 'create_task', and 'done' tools.
It chunks work effectively, creatively considers new features, and prunes irrelevant tasks.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console

import yaml
from loguru import logger

from glitchlab.history import TaskHistory
from glitchlab.router import Router
from glitchlab.controller import Task  # Import the strict Pydantic schema
from .scanner import Finding, ScanResult

console = Console()

AUDITOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Brainstorm new features, evaluate findings, and plan tasks to create or prune.",
            "parameters": {
                "type": "object",
                "properties": {
                    "evaluation": {"type": "string", "description": "Analysis of the findings and potential new features."},
                    "plan": {"type": "string", "description": "List of tasks you intend to create."}
                },
                "required": ["evaluation", "plan"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the workspace to get context.",
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
            "description": "Search the codebase for a pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "file_type": {"type": "string", "default": "*"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create and save a new GLITCHLAB task YAML.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique task ID (e.g. audit-feature-001)"},
                    "objective": {"type": "string", "description": "Clear, specific, actionable objective"},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "acceptance": {"type": "array", "items": {"type": "string"}},
                    "risk": {"type": "string", "enum": ["low", "medium", "high"]}
                },
                "required": ["id", "objective", "constraints", "acceptance", "risk"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that you have finished creating all necessary tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Summary of tasks created and findings pruned."}
                },
                "required": ["summary"]
            }
        }
    }
]

# Preserved for backward compatibility in case other scripts import it
def group_findings_into_tasks(result: ScanResult) -> list[list[Finding]]:
    """Legacy finding chunking"""
    return [result.findings]


def _check_task_complexity(objective: str, constraints: list) -> str | None:
    """
    Check if a task objective is too complex for a single implementation pass.

    Returns a rejection reason string if too complex, None if acceptable.
    """
    if isinstance(constraints, str):
        constraints = [constraints]
    combined = objective + " " + " ".join(constraints or [])

    issues = []

    # Check file count: more than 2 file paths mentioned
    file_refs = re.findall(r'[\w/]+\.\w{1,4}', combined)
    # Deduplicate
    unique_files = set(f for f in file_refs if not f.startswith("e.g"))
    if len(unique_files) > 2:
        issues.append(f"references {len(unique_files)} files ({', '.join(list(unique_files)[:4])})")

    # Check action count: more than 2 distinct action verbs
    action_verbs = ["add", "remove", "refactor", "update", "create", "delete",
                    "replace", "migrate", "rename", "rewrite", "extract", "move"]
    found_verbs = [v for v in action_verbs if re.search(rf'\b{v}\b', objective, re.IGNORECASE)]
    if len(found_verbs) > 2:
        issues.append(f"has {len(found_verbs)} action verbs ({', '.join(found_verbs)})")

    # Check for vague compound objectives
    if re.search(r'\band\b.*\band\b', objective, re.IGNORECASE):
        # Count the "and" clauses
        and_count = len(re.findall(r'\band\b', objective, re.IGNORECASE))
        if and_count >= 2:
            issues.append(f"has {and_count} 'and' conjunctions — likely multiple tasks combined")

    # Check for vague language
    vague_terms = []
    for term in ["clean up", "improve", "make better", "handle properly",
                 "update as needed", "as appropriate", "optimize", "overhaul"]:
        if term.lower() in objective.lower():
            vague_terms.append(term)
    if vague_terms:
        issues.append(f"uses vague language: {', '.join(vague_terms)}")

    if len(issues) >= 2:
        return "Task is too complex: " + "; ".join(issues) + "."

    # Single issue is a warning but not a rejection, unless it's vague language
    if vague_terms:
        return "Task uses vague language: " + ", ".join(vague_terms) + ". Be specific about the exact change."

    return None


class TaskWriter:
    """
    Generates GLITCHLAB task YAML files from scanner findings and ideation.
    Operates as an agentic loop to explore, plan, and write tasks.
    """

    def __init__(self, router: Router, output_dir: Path, dry_run: bool = False):
        self.router = router
        self.output_dir = output_dir
        self.dry_run = dry_run
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _gather_failure_context(self, repo_path: Path, max_failures: int = 10) -> str:
        """Read recent task failures from history for auditor context."""
        try:
            history = TaskHistory(repo_path)
            failures = history.get_failures(max_failures)
        except Exception:
            return ""

        if not failures:
            return ""

        lines = [
            "## Previous Task Failures (learn from these — do not regenerate equivalent tasks)"
        ]
        for f in failures:
            task_id = f.get("task_id", "unknown")
            status = f.get("status", "unknown")
            error = f.get("error", "N/A")
            lines.append(f"- **{task_id}** ({status}): {error}")
            events = f.get("events_summary", {})
            if events.get("security_verdict"):
                lines.append(f"  Security: {events['security_verdict']}")
            if events.get("fix_attempts"):
                lines.append(f"  Fix attempts: {events['fix_attempts']}")
        return "\n".join(lines)

    def _read_roadmap(self, repo_path: Path) -> str:
        """Read .glitchlab/ROADMAP.md for task prioritization context."""
        roadmap = repo_path / ".glitchlab" / "ROADMAP.md"
        if not roadmap.exists():
            return ""
        try:
            content = roadmap.read_text(encoding="utf-8", errors="ignore")[:4000]
            return f"## Project Roadmap\nPrioritize tasks that advance 'Now' items. Skip findings in 'Deferred' areas.\n\n{content}"
        except Exception:
            return ""

    def write_tasks(self, result: ScanResult) -> list[Path]:
        """Generate task YAML files for all findings using agent loop. Returns list of written paths."""
        written_paths: list[Path] = []
        
        # We only pass top 50 findings to avoid blowing up the context window
        findings_text = "\n".join(
            f"- [{f.kind}] {f.file}:{f.line} — {f.description}"
            for f in result.findings[:50]
        )
        if len(result.findings) > 50:
            findings_text += f"\n... and {len(result.findings) - 50} more."

        system_prompt = """You are the GLITCHLAB Auditor & Ideation Agent.
You operate in a tool-calling loop to generate high-quality development tasks.

Your responsibilities:
1. Evaluate static scanner findings. Prune irrelevant ones, and group the real issues into cohesive tasks.
2. Creatively consider NEW features, refactors, or architectural improvements that would benefit the codebase.
3. Break these ideas down into SMALL, actionable tasks. See sizing rules below.
4. Use `read_file` and `search_grep` to validate your ideas and understand the codebase before writing a task.
5. Use `create_task` to write each task.
6. Call `done` when finished.

## Task sizing rules (CRITICAL — tasks that violate these will be REJECTED)

- Each task MUST touch at most 2 files. If a change requires 3+ files, split into sequential subtasks.
- Each task MUST have ONE primary action: add, remove, refactor, create, rename, or fix. Not multiple.
- If a finding needs work across multiple files, create numbered sequential tasks:
  e.g., 'refactor-auth-001' (create the new module), 'refactor-auth-002' (update imports), 'refactor-auth-003' (add tests).
- Later tasks can note dependencies in constraints: "Depends on refactor-auth-001 being complete."
- NEVER use vague action words: "clean up", "improve", "make better", "handle properly", "update as needed".
  Instead be specific: "Add type hint `str` to parameter `name`", "Replace the inline dict on line 45 with a pytest fixture".
- Each task's objective must name the specific file(s) and the specific change.
- Aim for 8-16 tasks per scan. More small tasks is better than fewer large ones.

Make sure tasks are DEPENDABLE and HIGH QUALITY. Don't create vague tasks.
A good task tells the implementer exactly what files to touch and what behavior to achieve.
A bad task says "refactor X to be cleaner." A good task says "Extract the validation logic in X.validate() into a standalone function validate_input() in the same file."
"""

        roadmap_context = self._read_roadmap(result.repo_path)
        failure_context = self._gather_failure_context(result.repo_path)

        user_content = f"""Scanner Findings:
{findings_text if result.findings else "No static findings. Focus on ideating new features!"}

{roadmap_context}

{failure_context}

Repo: {result.repo_path}
Output Dir: {self.output_dir}

Plan your work, read necessary files, write the tasks, and call `done`.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        logger.info("[AUDITOR] Starting agentic task generation loop...")

        max_steps = 30
        think_count = 0

        for step in range(max_steps):
            logger.debug(f"[AUDITOR] Loop Step {step+1}/{max_steps}...")

            step_kwargs = {}
            if step == 0:
                step_kwargs["tool_choice"] = {"type": "function", "function": {"name": "think"}}

            try:
                response = self.router.complete(
                    role="auditor",
                    messages=messages,
                    tools=AUDITOR_TOOLS,
                    **step_kwargs
                )
            except Exception as e:
                logger.error(f"[AUDITOR] LLM generation failed: {e}")
                break

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
                messages.append({"role": "user", "content": "Please use your tools to create tasks or call `done`."})
                continue

            for tool_call in response.tool_calls:
                tc_name = tool_call.function.name
                tc_id = tool_call.id
                try:
                    tc_args = json.loads(tool_call.function.arguments or "{}")
                except Exception:
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": "Invalid JSON args"})
                    continue

                if tc_name == "think":
                    think_count += 1
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": "Thought logged. Proceed with reading files or creating tasks."})
                
                elif tc_name == "read_file":
                    path = result.repo_path / tc_args.get("path", "")
                    try:
                        content = path.read_text(encoding="utf-8", errors="ignore")[:3000] # Limit read size
                        res = f"Read from {path.name}:\n{content}"
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
                            "--exclude-dir=.git",
                            pattern, "."
                        ]
                        proc = subprocess.run(cmd, cwd=result.repo_path, capture_output=True, text=True, timeout=10)
                        out = proc.stdout if proc.stdout else "No matches found."
                        res = "\n".join(out.splitlines()[:50])
                    except Exception as e:
                        res = f"Search error: {e}"
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "create_task":
                    task_id = tc_args.get("id", "audit-001")
                    safe_id = re.sub(r"[^\w\-]", "-", task_id)
                    path = self.output_dir / f"{safe_id}.yaml"

                    # --- Complexity gate: reject oversized tasks ---
                    objective = tc_args.get("objective", "")
                    rejection = _check_task_complexity(objective, tc_args.get("constraints", []))
                    if rejection:
                        res = (
                            f"REJECTED: {rejection} "
                            f"Split this into smaller tasks, each touching at most 2 files "
                            f"with one action. Use sequential IDs (e.g., '{safe_id}-01', '{safe_id}-02'). "
                            f"Then call create_task for each subtask."
                        )
                        logger.info(f"[AUDITOR] Rejected oversized task '{safe_id}': {rejection}")
                        messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})
                        continue
                    # --- End complexity gate ---

                    try:
                        risk = tc_args.get("risk", "low")
                        if risk not in ("low", "medium", "high"):
                            risk = "medium"

                        constraints = tc_args.get("constraints", [])
                        task_data = {
                            "id": safe_id,
                            "objective": tc_args.get("objective", "Generated task"),
                            "constraints": constraints,
                            "acceptance": tc_args.get("acceptance", []),
                            "risk": risk,
                            "source": "auditor",
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                        }
                        # Validate via Pydantic model implicitly
                        Task(**task_data) 

                        if not self.dry_run:
                            with open(path, "w") as f:
                                yaml.dump(task_data, f, sort_keys=False)
                            written_paths.append(path)
                            res = f"Successfully created task at {path.name}"
                            console.print(f"  [bold green]📝 Created task:[/] {path.name}")
                        else:
                            written_paths.append(path)
                            res = f"Dry run: Simulated task creation at {path.name}"
                            console.print(f"  [bold green]📝 [DRY RUN] Would create task:[/] {path.name}")
                            console.print(f"    [dim]Objective: {task_data['objective']}[/]")
                            
                    except Exception as e:
                        res = f"Failed to create task: {e}"
                        console.print(f"  [red]❌ Task creation failed: {e}[/]")
                        
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "done":
                    logger.info(f"[AUDITOR] Agent finished: {tc_args.get('summary', '')}")
                    return written_paths

        logger.warning("[AUDITOR] Hit max steps without calling done.")
        return written_paths