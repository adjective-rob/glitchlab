"""
GLITCHLAB Controller — The Brainstem

The most important piece. It is NOT smart. It is deterministic.

Responsibilities:
  - Pull next task
  - Create isolated worktree
  - Assign agent roles in order
  - Track token budget
  - Track retry attempts
  - Enforce stop conditions
  - Open PR
  - Clean up workspace

It never writes code. It only coordinates.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table

from glitchlab.agents import AgentContext
from glitchlab.agents.archivist import ArchivistAgent
from glitchlab.agents.debugger import DebuggerAgent
from glitchlab.agents.implementer import ImplementerAgent
from glitchlab.agents.planner import PlannerAgent
from glitchlab.agents.release import ReleaseAgent
from glitchlab.agents.security import SecurityAgent
from glitchlab.config_loader import GlitchLabConfig, load_config
from glitchlab.governance import BoundaryEnforcer, BoundaryViolation
from glitchlab.history import TaskHistory
from glitchlab.indexer import build_index
from glitchlab.prelude import PreludeContext
from glitchlab.router import BudgetExceededError, Router
from glitchlab.workspace import Workspace
from glitchlab.workspace.tools import ToolExecutor, ToolViolationError

console = Console()


# ---------------------------------------------------------------------------
# Task Definition
# ---------------------------------------------------------------------------

class Task:
    """Represents a single unit of work for GLITCHLAB."""

    def __init__(
        self,
        task_id: str,
        objective: str,
        constraints: list[str] | None = None,
        acceptance_criteria: list[str] | None = None,
        risk_level: str = "low",
        source: str = "local",
        mode: str | None = None,
    ):
        self.task_id = task_id
        self.objective = objective
        self.constraints = constraints or []
        self.acceptance_criteria = acceptance_criteria or ["Tests pass", "Clean diff"]
        self.risk_level = risk_level
        self.source = source

        # Explicit Governance Mode
        if mode:
            self.mode = mode
        else:
            if self.risk_level == "low" and any(term in self.objective.lower() for term in ["doc", "lint", "format", "fix"]):
                self.mode = "maintenance"
            else:
                self.mode = "evolution"

    @classmethod
    def from_yaml(cls, path: Path) -> "Task":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            task_id=data.get("id", path.stem),
            objective=data["objective"],
            constraints=data.get("constraints", []),
            acceptance_criteria=data.get("acceptance", []),
            risk_level=data.get("risk", "low"),
            source="local-file",
            mode=data.get("mode"),
        )

    @classmethod
    def from_interactive(cls, objective: str) -> "Task":
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return cls(
            task_id=f"interactive-{ts}",
            objective=objective,
            source="interactive",
        )

    @classmethod
    def from_github_issue(cls, repo_path: Path, issue_number: int) -> "Task":
        """Fetch issue from GitHub CLI."""
        result = subprocess.run(
            ["gh", "issue", "view", str(issue_number), "--json", "title,body,labels,number"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to fetch issue #{issue_number}: {result.stderr}")

        data = json.loads(result.stdout)
        labels = [lbl["name"] for lbl in data.get("labels", [])]

        mode = "maintenance" if "maintenance" in labels else ("evolution" if "evolution" in labels else None)

        return cls(
            task_id=f"gh-{issue_number}",
            objective=f"{data['title']}\n\n{data.get('body', '')}",
            risk_level="high" if "core" in labels else "low",
            source="github",
            mode=mode,
        )


# ---------------------------------------------------------------------------
# File Context Builder
# ---------------------------------------------------------------------------

def gather_file_context(working_dir: Path, files: list[str], max_lines: int = 2000) -> dict[str, str]:
    """Read file contents for agent context. Truncates large files."""
    context = {}
    for fpath in files:
        full = working_dir / fpath
        if full.exists() and full.is_file():
            try:
                lines = full.read_text().splitlines()
                if len(lines) > max_lines:
                    content = "\n".join(lines[:max_lines]) + f"\n\n... truncated ({len(lines)} lines total)"
                else:
                    content = "\n".join(lines)
                context[fpath] = content
            except Exception as e:
                context[fpath] = f"(could not read: {e})"
    return context


# ---------------------------------------------------------------------------
# Change Applicator (supports full content + unified diffs)
# ---------------------------------------------------------------------------

def apply_changes(
    working_dir: Path,
    changes: list[dict],
    boundary: BoundaryEnforcer | None = None,
    allow_core: bool = False,
    allow_test_modifications: bool = False,
    allow_full_rewrite: bool = True,
) -> list[str]:
    """
    Apply implementation changes to the workspace.
    Enforces mode-based safety bounds at the lowest layer.
    """
    applied = []
    for change in changes:
        filename = change.get("file", "")
        if not filename:
            continue

        # 1. Strict Boundary Check
        if boundary:
            boundary.check([filename], allow_core)

        # 2. Strict Test Mutation Check
        is_test = any(term in filename.lower() for term in ["tests/", "test_", "_test", ".test.ts"])
        if is_test and not allow_test_modifications:
            raise BoundaryViolation(f"Test mutation not allowed in maintenance mode: {filename}")

        fpath = working_dir / filename
        action = change.get("action", "modify")

        if action == "create":
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(change.get("content", ""))
            applied.append(f"CREATE {filename}")

        elif action == "delete":
            if fpath.exists():
                fpath.unlink()
                applied.append(f"DELETE {filename}")

        elif action == "modify":
            patch = change.get("patch")
            file_content = change.get("content")

            if patch and patch.strip():
                success = _apply_patch(working_dir, patch)
                if success is True:
                    applied.append(f"PATCH {filename}")
                elif file_content:
                    # Patch failed but we have full content — use it
                    if not allow_full_rewrite and fpath.exists():
                        applied.append(f"FAIL {filename} (patch failed, full-file rewrite blocked in maintenance mode)")
                    else:
                        fpath.parent.mkdir(parents=True, exist_ok=True)
                        fpath.write_text(file_content)
                        applied.append(f"MODIFY {filename} (patch failed, used full content)")
                    # Log the patch error for debugging
                    if isinstance(success, str):
                        logger.info(f"[PATCH] Fell back to content for {filename}: {success}")
                else:
                    # Patch failed, no content fallback
                    error_detail = f": {success}" if isinstance(success, str) else ""
                    applied.append(f"FAIL {filename} (patch failed, no fallback{error_detail})")
            elif file_content:
                if not allow_full_rewrite and fpath.exists():
                    applied.append(f"FAIL {filename} (full-file rewrite blocked in maintenance mode)")
                else:
                    # Guard against hallucinated rewrites of large files
                    existing_lines = fpath.read_text().splitlines() if fpath.exists() else []
                    new_lines = file_content.splitlines()
                    if fpath.exists() and len(existing_lines) > 100 and len(new_lines) < len(existing_lines) * 0.7:
                        applied.append(f"FAIL {filename} (content rejected — likely truncated rewrite)")
                    else:
                        fpath.parent.mkdir(parents=True, exist_ok=True)
                        fpath.write_text(file_content)
                        applied.append(f"MODIFY {filename}")
            else:
                applied.append(f"SKIP {filename} (no content or patch provided)")

    return applied


def apply_tests(
    working_dir: Path,
    tests: list[dict],
    allow_test_modifications: bool = False,
) -> list[str]:
    """Apply test file changes with explicit permission check."""
    if tests and not allow_test_modifications:
        raise BoundaryViolation("Test mutation blocked by current governance mode.")

    applied = []
    for test in tests:
        fpath = working_dir / test["file"]
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(test.get("content", ""))
        applied.append(f"TEST {test['file']}")
    return applied


def _apply_patch(working_dir: Path, patch: str) -> bool | str:
    """
    Apply a unified diff using the 'patch' CLI.
    Returns True on success, or an error string on failure.
    """
    logger.debug(f"[PATCH] Raw patch content:\n{patch[:1000]}")

    # Strip markdown fences — #1 cause of "garbage in patch input"
    cleaned = patch.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    # Quick sanity check: does this look like a unified diff at all?
    if not any(line.startswith(("---", "diff ", "@@")) for line in cleaned.split("\n")):
        msg = "Not a valid unified diff (missing ---, diff, or @@ markers)"
        logger.warning(f"[PATCH] {msg}")
        return msg

    patch_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", dir=working_dir, delete=False
        ) as f:
            f.write(cleaned)
            patch_file = f.name

        result = subprocess.run(
            ["patch", "-p1", "--force", "--fuzz=3", "-i", patch_file],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            return True

        error = (result.stderr or result.stdout).strip()
        logger.warning(f"[PATCH] patch failed: {error}")
        return error

    except Exception as e:
        logger.warning(f"[PATCH] Exception applying patch: {e}")
        return str(e)
    finally:
        if patch_file:
            try:
                Path(patch_file).unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Surgical Doc Comment Insertion
# ---------------------------------------------------------------------------

def insert_doc_comments(file_path: Path, router: Any) -> bool:
    """
    Surgically insert /// doc comments above public functions that lack one.
    Asks the model only for comment text, does file manipulation in Python.
    Returns True if any changes were made.
    """
    lines = file_path.read_text().splitlines()
    functions_needing_docs = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("pub fn ") or stripped.startswith("pub async fn "):
            j = i - 1
            while j >= 0 and lines[j].strip() == "":
                j -= 1
            if j < 0 or not lines[j].strip().startswith("///"):
                functions_needing_docs.append((i, line))

    if not functions_needing_docs:
        logger.info("[DOC] No public functions missing doc comments.")
        return False

    fn_list = "\n".join(
        f"Line {i+1}: {line.strip()}" for i, line in functions_needing_docs
    )
    prompt = f"""For each of the following Rust public functions, write a single concise /// doc comment (one line only).
Return a JSON array where each item has "line" (the line number) and "comment" (the full /// comment string).
Example: [{{"line": 42, "comment": "/// Initializes the vault and loads existing keys."}}]

Functions:
{fn_list}
"""
    response = router.complete(
        role="implementer",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )

    content = response.content.strip()
    if content.startswith("```"):
        content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))

    try:
        comments = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                comments = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("[DOC] Failed to parse doc comments from model response")
                return False
        else:
            logger.error("[DOC] Failed to parse doc comments from model response")
            return False

    comment_map = {item["line"]: item["comment"] for item in comments}

    # Insert in reverse order to preserve line numbers
    for i, line in reversed(functions_needing_docs):
        line_num = i + 1
        comment = comment_map.get(line_num)
        if comment:
            indent = len(line) - len(line.lstrip())
            comment_text = comment.strip()
            if not comment_text.startswith("///"):
                comment_text = "/// " + comment_text
            comment_line = " " * indent + comment_text
            lines.insert(i, comment_line)

    file_path.write_text("\n".join(lines) + "\n")
    logger.info(f"[DOC] Inserted {len(comment_map)} doc comments into {file_path.name}")
    return True


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class Controller:
    """
    The GLITCHLAB brainstem.

    Deterministic orchestration of the agent pipeline:
      Plan → Implement → Test → Debug Loop → Security → Release → PR
    """

    def __init__(
        self,
        repo_path: Path,
        config: GlitchLabConfig | None = None,
        allow_core: bool = False,
        auto_approve: bool = False,
        test_command: str | None = None,
    ):
        self.repo_path = repo_path.resolve()
        self.config = config or load_config(repo_path)
        self.allow_core = allow_core
        self.auto_approve = auto_approve
        self.test_command = test_command

        # Initialize components
        self.router = Router(self.config)
        self.boundary = BoundaryEnforcer(self.config.boundaries.protected_paths)

        # Initialize agents
        self.planner = PlannerAgent(self.router)
        self.implementer = ImplementerAgent(self.router)
        self.debugger = DebuggerAgent(self.router)
        self.security = SecurityAgent(self.router)
        self.release = ReleaseAgent(self.router)
        self.archivist = ArchivistAgent(self.router)

        # Task state
        self._task: Task | None = None
        self._workspace: Workspace | None = None
        self._event_log: list[dict] = []
        self._repo_index_context: str = ""

        # History tracking
        self._history = TaskHistory(self.repo_path)

        # Prelude integration — codebase context
        self._prelude = PreludeContext(self.repo_path)
        self._prelude_prefix: str = ""

    def run(self, task: Task) -> dict[str, Any]:
        """Execute the full agent pipeline for a task."""
        self._task = task
        result: dict[str, Any] = {
            "task_id": task.task_id,
            "status": "pending",
            "events": [],
        }

        console.print(Panel(
            f"[bold green]Task:[/] {task.objective[:120]}\n"
            f"[bold]ID:[/] {task.task_id}  |  [bold]Source:[/] {task.source}\n"
            f"[bold]Risk:[/] {task.risk_level}  |  [bold]Mode:[/] {task.mode.upper()}",
            title="⚡ GLITCHLAB",
            subtitle="Build Weird. Ship Clean.",
            border_style="bright_green",
        ))

        try:
            # 1. Create workspace
            self._workspace = Workspace(
                self.repo_path, task.task_id,
                self.config.workspace.worktree_dir,
            )
            ws_path = self._workspace.create()
            self._log_event("workspace_created", {"path": str(ws_path)})

            tools = ToolExecutor(
                allowed_tools=self.config.allowed_tools,
                blocked_patterns=self.config.blocked_patterns,
                working_dir=ws_path,
            )

            # 1.5. Gather Prelude context (codebase memory)
            if self._prelude.available:
                console.print("\n[bold dim]📋 [PRELUDE] Loading project context...[/]")
                self._prelude.refresh()
                self._prelude_prefix = self._prelude.build_agent_prefix()
                if self._prelude_prefix:
                    summary = self._prelude.summary()
                    console.print(
                        f"  [dim]Context loaded: {len(summary.get('files', []))} files, "
                        f"{summary.get('decisions_count', 0)} ADRs[/]"
                    )
                    self._log_event("prelude_loaded", summary)

                    # Merge Prelude constraints into task constraints
                    prelude_constraints = self._prelude.get_constraints()
                    if prelude_constraints:
                        task.constraints = list(set(task.constraints + prelude_constraints))

            # 1.6. Build repo index (file discovery)
            console.print("\n[bold dim]🗂  [INDEX] Scanning repository...[/]")
            repo_index = build_index(ws_path)
            self._repo_index_context = repo_index.to_agent_context(max_files=200)
            console.print(
                f"  [dim]{repo_index.total_files} files, "
                f"{len(repo_index.crates)} crates, "
                f"{len(repo_index.languages)} languages[/]"
            )
            self._log_event("repo_indexed", {
                "total_files": repo_index.total_files,
                "languages": repo_index.languages,
                "crates": repo_index.crates,
            })

            # 1.7. Load failure context from history
            failure_context = self._history.build_failure_context()
            if failure_context:
                console.print("  [dim]Loaded recent failure patterns for planner[/]")

            # 2. Plan
            plan = self._run_planner(task, ws_path, failure_context)
            if plan.get("parse_error"):
                result["status"] = "plan_failed"
                return result

            # ------------------------------------------------------------
            # 3. Boundary Validation (Plan-Level)
            # ------------------------------------------------------------
            try:
                violations = self.boundary.check_plan(plan, self.allow_core)
                if violations:
                    self._log_event("core_override", {"files": violations})
                    console.print(f"[yellow]⚠ Core override granted for: {violations}[/]")
            except BoundaryViolation as e:
                console.print(f"[red]🚫 {e}[/]")
                result["status"] = "boundary_violation"
                return result

            # ------------------------------------------------------------
            # 4. Governance Mode Routing
            # ------------------------------------------------------------
            is_maintenance = task.mode == "maintenance"
            is_evolution = task.mode == "evolution"

            if is_evolution:
                # Evolution mode always requires human gate before PR
                self.config.intervention.pause_before_pr = True

            # Strict doc-only detection
            objective_lower = task.objective.lower()
            is_doc_only = (
                is_maintenance
                and any(term in objective_lower for term in ["doc", "documentation", "///"])
                and task.risk_level == "low"
                and all(step.get("action") == "modify" for step in plan.get("steps", []))
                and any(f.endswith(".rs") for f in plan.get("files_likely_affected", []))
            )

            # ------------------------------------------------------------
            # 4A. Maintenance: Surgical Documentation Path
            # ------------------------------------------------------------
            if is_doc_only:
                console.print(
                    "\n[bold dim]📄 [MAINTENANCE MODE] "
                    "Surgical documentation update — implementer bypassed.[/]"
                )

                impl = {
                    "changes": [],
                    "tests_added": [],
                    "commit_message": f"docs: update documentation for {task.task_id}",
                    "summary": "Surgical documentation insertion.",
                }

                applied = []

                for f in plan.get("files_likely_affected", []):
                    fpath = ws_path / f
                    if fpath.exists():
                        inserted = insert_doc_comments(fpath, self.router)
                        if inserted:
                            applied.append(f"DOC {f}")

                for entry in applied:
                    console.print(f"  [cyan]{entry}[/]")

            # ------------------------------------------------------------
            # 4B. Standard Execution Path (Maintenance or Evolution)
            # ------------------------------------------------------------
            else:
                impl = self._run_implementer(task, plan, ws_path)

                if impl.get("parse_error"):
                    result["status"] = "implementation_failed"
                    return result

                # Complexity Heuristic: If large refactor, allow full rewrite
                is_high_complexity = plan.get("estimated_complexity", "").lower() in ["high", "large"]
                should_allow_rewrite = (not is_maintenance) or is_high_complexity

                if is_high_complexity:
                    console.print("  [dim]High complexity detected: allowing full-file rewrites to ensure diff stability.[/]")

                # Enforce mode-based write constraints at lowest layer
                try:
                    applied = apply_changes(
                        ws_path,
                        impl.get("changes", []),
                        boundary=self.boundary,
                        allow_core=self.allow_core,
                        allow_test_modifications=not is_maintenance,
                        allow_full_rewrite=should_allow_rewrite,
                    )

                    applied += apply_tests(
                        ws_path,
                        impl.get("tests_added", []),
                        allow_test_modifications=not is_maintenance,
                    )

                except BoundaryViolation as e:
                    console.print(f"[red]🚫 Boundary Violation during implementation: {e}[/]")
                    result["status"] = "boundary_violation"
                    return result

                for entry in applied:
                    console.print(f"  [cyan]{entry}[/]")

                if any("FAIL" in a for a in applied):
                    console.print("[red]❌ Patches failed to apply. Aborting to prevent empty PR.[/]")
                    result["status"] = "patch_failed"
                    return result

            # ----------------------------------------
            # Patch failure retry (one attempt)
            # ----------------------------------------
            patch_failures = [a for a in applied if a.startswith("PATCH_FAILED")]

            if patch_failures:
                console.print("[yellow]⚠ Patch failed. Attempting one auto-repair...[/]")

                retry_impl = self._retry_patch(
                    task,
                    plan,
                    ws_path,
                    impl,
                    applied,
                )

                if retry_impl.get("parse_error"):
                    result["status"] = "implementation_failed"
                    return result

                applied = apply_changes(
                    ws_path,
                    retry_impl.get("changes", []),
                    boundary=self.boundary,
                    allow_core=self.allow_core,
                    allow_test_modifications=not is_maintenance,
                    allow_full_rewrite=not is_maintenance,
                )

                for entry in applied:
                    console.print(f"  [cyan]{entry}[/]")

                if any(a.startswith("PATCH_FAILED") for a in applied):
                    console.print("[red]❌ Patch retry failed. Aborting.[/]")
                    result["status"] = "implementation_failed"
                    return result

            # Define standard bypass structures if skipping phases
            if is_doc_only:
                test_ok = True
                sec = {"verdict": "pass", "issues": []}
                rel = {
                    "version_bump": "none",
                    "reasoning": "Maintenance mode — documentation only",
                    "changelog_entry": "- Documentation updates",
                }
            else:
                # --------------------------------------------------------
                # 5. Test + Debug Loop
                # --------------------------------------------------------
                if self.test_command:
                    test_ok = self._run_fix_loop(task, ws_path, tools, impl)

                    if not test_ok:
                        result["status"] = "tests_failed"
                        console.print("[red]❌ Fix loop exhausted. Tests still failing.[/]")

                        if not self._confirm("Continue to PR anyway?"):
                            return result
                else:
                    test_ok = True

                # --------------------------------------------------------
                # 6. Security Review
                # --------------------------------------------------------
                sec = self._run_security(task, impl, ws_path)

                if sec.get("verdict") == "block":
                    console.print("[red]🚫 Security blocked this change.[/]")
                    self._print_security_issues(sec)

                    if not self._confirm("Override security block?"):
                        result["status"] = "security_blocked"
                        return result

                # --------------------------------------------------------
                # 7. Release Assessment
                # --------------------------------------------------------
                rel = self._run_release(task, impl, ws_path)

                # ------------------------------------------------------------
                # 7.5. Archivist (Governed Documentation)
                # ------------------------------------------------------------
                nova_result = self._run_archivist(task, impl, plan, rel, ws_path)

                # Maintenance mode never auto-writes ADRs
                if is_maintenance:
                    nova_result["should_write_adr"] = False

                if nova_result.get("should_write_adr") and nova_result.get("adr"):
                    adr_applied = self._write_adr(ws_path, nova_result["adr"])
                    if adr_applied:
                        console.print(f"  [cyan]{adr_applied}[/]")

                for doc_update in nova_result.get("doc_updates", []):
                    doc_applied = self._write_doc_update(ws_path, doc_update)
                    if doc_applied:
                        console.print(f"  [cyan]{doc_applied}[/]")

                # Maintenance mode: forbid file create/delete and out-of-scope edits before commit
                if is_maintenance:
                    allowed = set(plan.get("files_likely_affected") or [])
                    if not allowed:
                        raise RuntimeError(
                            "Maintenance mode requires explicit files_likely_affected"
                        )

                    # Stage changes produced by this task
                    for path in allowed:
                        self._workspace._git("add", path, check=False)

                    # Inspect ONLY staged changes
                    diff_output = self._workspace._git(
                        "diff", "--cached", "--name-status", check=False
                    )
                    lines = diff_output.splitlines() if diff_output else []

                    created, deleted, touched = [], [], []
                    for line in lines:
                        parts = line.split("\t", 1)
                        if len(parts) != 2:
                            continue
                        status, path = parts

                        if status == "A":
                            created.append(path)
                        elif status == "D":
                            deleted.append(path)
                        else:
                            touched.append(path)

                    out_of_scope = [p for p in touched if p not in allowed]

                    if created or deleted or out_of_scope:
                        raise RuntimeError(
                            f"Maintenance violation. "
                            f"created={created} deleted={deleted} "
                            f"out_of_scope={out_of_scope}"
                        )

            # 8. Commit + PR
            commit_msg = impl.get("commit_message", f"glitchlab: {task.task_id}")
            self._workspace.commit(commit_msg)

            if self.config.intervention.pause_before_pr:
                diff = self._workspace.diff_stat()
                console.print(Panel(diff, title="Diff Summary", border_style="cyan"))
                if not self._confirm("Create PR?"):
                    result["status"] = "pr_cancelled"
                    return result

            # Push + PR
            try:
                self._workspace.push()
                pr_url = self._create_pr(task, impl, rel)
                result["pr_url"] = pr_url
                result["status"] = "pr_created"
                console.print(f"[bold green]✅ PR created: {pr_url}[/]")
            except Exception as e:
                logger.warning(f"PR creation failed: {e}")
                result["status"] = "committed"
                result["branch"] = self._workspace.branch_name
                console.print(f"[yellow]Changes committed to {self._workspace.branch_name}[/]")

            # Budget summary
            self._print_budget_summary()

            result["events"] = self._event_log
            result["budget"] = self.router.budget.summary()

        except BudgetExceededError as e:
            console.print(f"[red]💸 Budget exceeded: {e}[/]")
            result["status"] = "budget_exceeded"
        except KeyboardInterrupt:
            console.print("\n[yellow]⚡ Interrupted by human.[/]")
            result["status"] = "interrupted"
        except Exception as e:
            logger.exception("Controller error")
            console.print(f"[red]💥 Error: {e}[/]")
            result["status"] = "error"
            result["error"] = str(e)
        finally:
            if self._workspace:
                try:
                    self._workspace.cleanup()
                except Exception:
                    pass

            # Always record to history
            self._history.record(result)

        return result

    # -----------------------------------------------------------------------
    # Agent Runners
    # -----------------------------------------------------------------------

    def _run_planner(self, task: Task, ws_path: Path, failure_context: str = "") -> dict:
        console.print("\n[bold magenta]🧠 [ZAP] Planning...[/]")

        # Build rich objective with all available context
        objective_parts = []

        if self._prelude_prefix:
            objective_parts.append(self._prelude_prefix)

        if self._repo_index_context:
            objective_parts.append(self._repo_index_context)

        if failure_context:
            objective_parts.append(failure_context)

        objective_parts.append(f"TASK:\n{task.objective}")

        objective = "\n\n---\n\n".join(objective_parts)

        context = AgentContext(
            task_id=task.task_id,
            objective=objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            constraints=task.constraints,
            acceptance_criteria=task.acceptance_criteria,
            risk_level=task.risk_level,
        )

        plan = self.planner.run(context)
        self._log_event("plan_created", {
            "steps": len(plan.get("steps", [])),
            "risk": plan.get("risk_level"),
        })

        # Display plan
        self._print_plan(plan)

        # Human gate
        if self.config.intervention.pause_after_plan and not self.auto_approve:
            if not self._confirm("Approve plan?"):
                plan["_aborted"] = True
                plan["parse_error"] = True
                return plan

        return plan

    def _run_implementer(self, task: Task, plan: dict, ws_path: Path) -> dict:
        console.print("\n[bold blue]🔧 [PATCH] Implementing...[/]")

        file_context = gather_file_context(
            ws_path,
            plan.get("files_likely_affected", []),
        )

        # Include Prelude context so Patch knows the project's patterns
        extra = {}
        if self._prelude_prefix:
            extra["prelude_context"] = self._prelude_prefix

        # ── FIX 4: Inject format guidance for reliable file operations ──
        impl_constraints = list(task.constraints)

        create_files = [
            s.get("files", []) for s in plan.get("steps", [])
            if s.get("action") == "create"
        ]
        if any(create_files):
            impl_constraints.append(
                "For NEW files (action='create'), always provide complete file "
                "content in the 'content' field. Never use unified diffs for new files."
            )

        impl_constraints.append(
            "For MODIFIED files (action='modify'), prefer providing complete file "
            "content in the 'content' field. Only use the 'patch' field if the file "
            "is large (>200 lines) and the change is small. Never wrap patches in "
            "markdown code fences."
        )

        context = AgentContext(
            task_id=task.task_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            constraints=impl_constraints,
            acceptance_criteria=task.acceptance_criteria,
            file_context=file_context,
            previous_output=plan,
            extra=extra,
        )

        impl = self.implementer.run(context, max_tokens=8192)

        # For doc-comment tasks, use surgical insertion instead of full rewrite
        for change in impl.get("changes", []):
            if change.get("action") == "modify":
                fpath = ws_path / change["file"]
                if fpath.exists():
                    inserted = insert_doc_comments(fpath, self.router)
                    if inserted:
                        change["_doc_inserted"] = True
                        change["content"] = None  # prevent full rewrite fallback

        self._log_event("implementation_created", {
            "changes": len(impl.get("changes", [])),
            "tests": len(impl.get("tests_added", [])),
        })

        return impl

    def _retry_patch(
        self,
        task: Task,
        plan: dict,
        ws_path: Path,
        original_impl: dict,
        applied_entries: list[str],
    ) -> dict:
        console.print("[dim]Re-prompting implementer with git error context...[/]")

        # Extract git error message
        error_lines = [
            a.replace("PATCH_ERROR ", "")
            for a in applied_entries
            if a.startswith("PATCH_ERROR")
        ]
        git_error = "\n".join(error_lines)

        file_context = gather_file_context(
            ws_path,
            plan.get("files_likely_affected", []),
        )

        retry_prompt = f"""
    The previous unified diff failed to apply with the following git error:

    {git_error}

    Regenerate a corrected unified diff that will apply cleanly.
    Ensure:
    - At least 3 context lines
    - Exact whitespace match
    - Valid unified diff format
    - No unrelated changes
    """

        context = AgentContext(
            task_id=task.task_id,
            objective=task.objective + "\n\n" + retry_prompt,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            constraints=task.constraints,
            acceptance_criteria=task.acceptance_criteria,
            file_context=file_context,
            previous_output=plan,
        )

        return self.implementer.run(context, max_tokens=8192)

    def _run_fix_loop(
        self, task: Task, ws_path: Path, tools: ToolExecutor, impl: dict
    ) -> bool:
        """Run test → debug → fix loop. Returns True if tests pass."""
        max_attempts = self.config.limits.max_fix_attempts
        previous_fixes = []

        for attempt in range(1, max_attempts + 1):
            console.print(f"\n[bold]🧪 Test run {attempt}/{max_attempts}...[/]")

            try:
                result = tools.execute(self.test_command)
            except ToolViolationError as e:
                console.print(f"[red]Tool violation: {e}[/]")
                return False

            if result.success:
                console.print("[green]✅ Tests pass![/]")
                self._log_event("tests_passed", {"attempt": attempt})
                return True

            error_output = result.stderr or result.stdout
            console.print(f"[red]❌ Tests failed (attempt {attempt})[/]")
            self._log_event("tests_failed", {"attempt": attempt})

            if attempt >= max_attempts:
                break

            # Invoke debugger
            console.print(f"\n[bold yellow]🐛 [REROUTE] Debugging (attempt {attempt})...[/]")

            file_context = gather_file_context(
                ws_path,
                [c["file"] for c in impl.get("changes", [])],
            )

            # ── FIX 3: Detect prior patch failures and tell LLM to use content ──
            patch_failed = any(
                "FAIL" in str(fix.get("_apply_result", ""))
                or "PATCH_FAILED" in str(fix.get("_apply_result", ""))
                for fix in previous_fixes
            )

            extra = {
                "error_output": error_output[:3000],
                "test_command": self.test_command,
                "attempt": attempt,
                "previous_fixes": previous_fixes,
            }

            if patch_failed or attempt > 1:
                extra["patch_strategy"] = (
                    "IMPORTANT: Previous unified diff patches failed to apply. "
                    "Do NOT output unified diffs. Instead, provide the COMPLETE "
                    "file content in the 'content' field of each change. "
                    "Set 'action' to 'modify' and leave 'patch' empty."
                )

            context = AgentContext(
                task_id=task.task_id,
                objective=task.objective,
                repo_path=str(self.repo_path),
                working_dir=str(ws_path),
                file_context=file_context,
                extra=extra,
            )

            debug_result = self.debugger.run(context)
            previous_fixes.append(debug_result)

            if not debug_result.get("should_retry", False):
                console.print("[yellow]Debugger says: don't retry.[/]")
                break

            # Apply fix
            fix_changes = debug_result.get("fix", {}).get("changes", [])
            if fix_changes:
                is_maintenance = (task.mode == "maintenance")
                applied = apply_changes(
                    ws_path,
                    fix_changes,
                    boundary=self.boundary,
                    allow_core=self.allow_core,
                    allow_test_modifications=(not is_maintenance),
                    allow_full_rewrite=True,
                )
                for a in applied:
                    console.print(f"  [cyan]{a}[/]")

                # ── FIX 2: Tag apply result and skip re-test if all patches failed ──
                debug_result["_apply_result"] = applied

                if all("FAIL" in a or "PATCH_FAILED" in a or "SKIP" in a for a in applied):
                    console.print("[yellow]⚠ Debug fix failed to apply. Skipping re-test.[/]")
                    continue
            else:
                console.print("[yellow]⚠ Debugger returned no fix changes.[/]")
                continue

        return False

    def _run_security(self, task: Task, impl: dict, ws_path: Path) -> dict:
        console.print("\n[bold red]🔒 [FRANKIE] Security scan...[/]")

        diff = self._workspace.diff_full() if self._workspace else ""

        context = AgentContext(
            task_id=task.task_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            previous_output=impl,
            extra={
                "diff": diff,
                "protected_paths": self.config.boundaries.protected_paths,
            },
        )

        result = self.security.run(context)
        self._log_event("security_review", {"verdict": result.get("verdict")})
        return result

    def _run_release(self, task: Task, impl: dict, ws_path: Path) -> dict:
        console.print("\n[bold cyan]📦 [SEMVER] Release assessment...[/]")

        diff = self._workspace.diff_stat() if self._workspace else ""

        context = AgentContext(
            task_id=task.task_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            previous_output=impl,
            extra={"diff": diff},
        )

        result = self.release.run(context)
        self._log_event("release_assessment", {
            "bump": result.get("version_bump"),
        })
        return result

    def _run_archivist(
        self, task: Task, impl: dict, plan: dict, release: dict, ws_path: Path
    ) -> dict:
        """Run Archivist Nova to document the change."""
        console.print("\n[bold dim]📚 [NOVA] Documenting...[/]")

        # Find existing doc files for context
        existing_docs = []
        for pattern in ["*.md", "docs/**/*.md", "doc/**/*.md"]:
            existing_docs.extend(
                str(p.relative_to(ws_path))
                for p in ws_path.glob(pattern)
                if p.is_file() and ".glitchlab" not in str(p)
            )

        context = AgentContext(
            task_id=task.task_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            previous_output=impl,
            extra={
                "plan": plan,
                "release": release,
                "prelude_context": self._prelude_prefix,
                "existing_docs": existing_docs[:20],
            },
        )

        result = self.archivist.run(context)
        self._log_event("archivist_completed", {
            "wrote_adr": result.get("should_write_adr", False),
            "doc_updates": len(result.get("doc_updates", [])),
        })
        return result

    @staticmethod
    def _write_adr(ws_path: Path, adr: dict) -> str | None:
        """Write an ADR to the workspace."""
        # Determine ADR directory — prefer .context/decisions/ for Prelude compat
        adr_dir = ws_path / ".context" / "decisions"
        if not adr_dir.exists():
            adr_dir = ws_path / "docs" / "adr"
        adr_dir.mkdir(parents=True, exist_ok=True)

        # Find next ADR number
        existing = list(adr_dir.glob("*.md"))
        next_num = len(existing) + 1

        title = adr.get("title", f"ADR-{next_num:03d}")
        safe_title = re.sub(r'[^a-z0-9\-]+', '-', title.lower())
        safe_title = re.sub(r'-+', '-', safe_title).strip('-')
        filename = f"{next_num:03d}-{safe_title[:50]}.md"
        filepath = adr_dir / filename

        content = f"""# {title}

**Status:** {adr.get('status', 'accepted')}
**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

## Context

{adr.get('context', 'N/A')}

## Decision

{adr.get('decision', 'N/A')}

## Consequences

{adr.get('consequences', 'N/A')}
"""
        alternatives = adr.get("alternatives_considered", [])
        if alternatives:
            content += "\n## Alternatives Considered\n\n"
            for alt in alternatives:
                content += f"- {alt}\n"

        content += "\n---\n*Generated by GLITCHLAB / Archivist Nova*\n"

        filepath.write_text(content)
        return f"ADR {filepath.relative_to(ws_path)}"

    @staticmethod
    def _write_doc_update(ws_path: Path, doc: dict) -> str | None:
        """Apply a documentation update."""
        fpath = ws_path / doc["file"]
        action = doc.get("action", "create")
        content = doc.get("content", "")

        if not content:
            return None

        if action == "create":
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)
            return f"DOC CREATE {doc['file']}"
        elif action == "append":
            fpath.parent.mkdir(parents=True, exist_ok=True)
            existing = fpath.read_text() if fpath.exists() else ""
            fpath.write_text(existing + "\n\n" + content)
            return f"DOC APPEND {doc['file']}"
        elif action == "update" and fpath.exists():
            fpath.write_text(content)
            return f"DOC UPDATE {doc['file']}"
        return None

    # -----------------------------------------------------------------------
    # PR Creation
    # -----------------------------------------------------------------------

    def _create_pr(self, task: Task, impl: dict, release: dict) -> str:
        """Create a GitHub PR via gh CLI."""
        title = impl.get("commit_message", f"glitchlab: {task.task_id}")
        body = self._build_pr_body(task, impl, release)

        result = subprocess.run(
            [
                "gh", "pr", "create",
                "--title", title,
                "--body", body,
                "--head", self._workspace.branch_name,
            ],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"PR creation failed: {result.stderr}")

        return result.stdout.strip()

    @staticmethod
    def _build_pr_body(task: Task, impl: dict, release: dict) -> str:
        body = f"""## 🔬 GLITCHLAB Automated PR

**Task:** {task.task_id}
**Source:** {task.source}

### Summary
{impl.get('summary', 'No summary provided.')}

### Changes
"""
        for change in impl.get("changes", []):
            body += f"- `{change.get('action', '?')}` {change.get('file', '?')}: {change.get('description', '')}\n"

        body += f"""
### Version Impact
- **Bump:** {release.get('version_bump', 'unknown')}
- **Reasoning:** {release.get('reasoning', 'N/A')}

### Changelog
{release.get('changelog_entry', 'N/A')}

---
*Generated by GLITCHLAB — Build Weird. Ship Clean.*
"""
        return body

    # -----------------------------------------------------------------------
    # Display Helpers
    # -----------------------------------------------------------------------

    def _print_plan(self, plan: dict) -> None:
        table = Table(title="Execution Plan", border_style="magenta")
        table.add_column("#", style="dim")
        table.add_column("Action")
        table.add_column("Files")
        table.add_column("Description")

        for step in plan.get("steps", []):
            table.add_row(
                str(step.get("step_number", "?")),
                step.get("action", "?"),
                ", ".join(step.get("files", [])),
                step.get("description", ""),
            )

        console.print(table)
        console.print(
            f"Risk: [bold]{plan.get('risk_level', '?')}[/] | "
            f"Core change: {plan.get('requires_core_change', False)} | "
            f"Complexity: {plan.get('estimated_complexity', '?')}"
        )

    def _print_security_issues(self, sec: dict) -> None:
        for issue in sec.get("issues", []):
            sev = issue.get("severity", "info")
            color = {"critical": "red", "high": "red", "medium": "yellow"}.get(sev, "dim")
            console.print(f"  [{color}]{sev.upper()}[/] {issue.get('description', '')}")

    def _print_budget_summary(self) -> None:
        summary = self.router.budget.summary()
        console.print(Panel(
            f"Tokens: {summary['total_tokens']:,} / "
            f"Cost: ${summary['estimated_cost']:.4f} / "
            f"Calls: {summary['call_count']}",
            title="💸 Budget",
            border_style="green",
        ))

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _confirm(self, prompt: str) -> bool:
        if self.auto_approve:
            return True
        return Confirm.ask(f"[bold]{prompt}[/]")

    def _log_event(self, event_type: str, data: dict | None = None) -> None:
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": self._task.task_id if self._task else None,
            "data": data or {},
        }
        self._event_log.append(event)
        logger.debug(f"[EVENT] {event_type}: {data}")