"""
GLITCHLAB Controller — The Brainstem (v2)

The most important piece. It is NOT smart. It is deterministic.

v2 Architecture: Context Router (not Context Reservoir)
  - Layer 0: Immutable system contracts per agent
  - Layer 1: ScopeResolver computes precise file/symbol context
  - Layer 2: Tool-centric retrieval (agents pull, not pushed)
  - Layer 3: Structured TaskState flows between agents

Responsibilities:
  - Pull next task
  - Create isolated worktree
  - Maintain TaskState across agent steps
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
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from glitchlab.agents import AgentContext, BaseAgent, AgentResult
from glitchlab.controller_utils import (
    attest_controller_action,
    calculate_quality_score,
    pre_task_git_fetch,
)
from glitchlab.registry import AGENT_REGISTRY, get_agent
from glitchlab.config_loader import GlitchLabConfig, PipelineStep, load_config
from glitchlab.display import build_pr_body, print_budget_summary, print_plan, print_security_issues
from glitchlab.doc_inserter import insert_doc_comments, write_adr
from glitchlab.runners import (
    run_archivist,
    run_auditor,
    run_delegated_agent,
    run_release,
    run_security,
)
from glitchlab.governance import BoundaryEnforcer, BoundaryViolation
from glitchlab.history import TaskHistory, extract_patterns_from_messages
from glitchlab.indexer import build_index
from glitchlab.prelude import PreludeContext
from glitchlab.router import BudgetExceededError, Router
from glitchlab.workspace import Workspace
from glitchlab.workspace.tools import ToolExecutor, ToolViolationError
from glitchlab.symbols import SymbolIndex
from glitchlab.task import Task, apply_changes, apply_tests
from glitchlab.event_bus import bus
from glitchlab.scope import ScopeResolver
from glitchlab.task_state import TaskState, StepState, DirtyRepoError  # re-export

console = Console()


# ---------------------------------------------------------------------------
# Controller (v2)
# ---------------------------------------------------------------------------

class Controller:
    """
    The GLITCHLAB brainstem (v2).

    Context Router architecture:
      - Agents receive only what they need via TaskState.to_agent_summary()
      - ScopeResolver computes file context from imports/AST, not planner guesses
      - Prelude is available as a tool, not injected globally
      - TaskState persists structured memory across agent steps

    Pipeline: Plan → Implement → Test → Debug Loop → Security → Release → PR
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

        # Core components
        self.router = Router(self.config)
        self.boundary = BoundaryEnforcer(self.config.boundaries.protected_paths)

        # Agents — instantiated from the central registry
        self.agents: dict[str, BaseAgent] = {
            role: get_agent(role, self.router)
            for role in AGENT_REGISTRY
        }

        # Run state (reset per-task)
        self._state: TaskState | None = None
        self._workspace: Workspace | None = None
        self._scope: ScopeResolver | None = None
        self._repo_index: Any = None
        self._repo_index_context: str = ""

        # History tracking
        self._history = TaskHistory(self.repo_path)

        # Prelude — available as tool context, NOT global prefix
        self._prelude = PreludeContext(self.repo_path)

    def _print_banner(self, task: Task) -> None:
        """Print the task banner."""
        console.print(Panel(
            f"[bold green]Task:[/] {task.objective[:120]}\n"
            f"[bold]ID:[/] {task.task_id}  |  [bold]Source:[/] {task.source}\n"
            f"[bold]Risk:[/] {task.risk_level}  |  [bold]Mode:[/] {task.mode.upper()}",
            title="⚡ GLITCHLAB v4.3.1",
            subtitle="Build Weird. Ship Clean.",
            border_style="bright_green",
        ))

    def _check_repo_clean(self) -> None:
        """Raise DirtyRepoError if the repo has uncommitted changes or is behind remote."""
        # --- EXECUTION GUARD (Manual Patch) ---
        # Check for uncommitted changes in the main repo, ignoring .glitchlab/
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        ).stdout

        # Filter out changes that are ONLY inside .glitchlab/ (logs, tasks, etc.)
        dirty_files = [
            line for line in status.splitlines()
            if not line[3:].startswith(".glitchlab/")
        ]

        if dirty_files:
            console.print("[red]🚫 Cannot run: Main repository has uncommitted changes:[/]")
            for f in dirty_files[:5]:
                console.print(f"  [dim]{f}[/]")
            raise DirtyRepoError("Clean your repository before running GLITCHLAB.")

        # Check if local branch is behind remote
        try:
            subprocess.run(["git", "fetch", "--quiet"], cwd=self.repo_path, timeout=10)
            behind = subprocess.run(
                ["git", "rev-list", "HEAD..@{u}", "--count"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            ).stdout.strip()
            if behind and behind.isdigit() and int(behind) > 0:
                console.print(f"[red]🚫 Cannot run: Local branch is behind remote by {behind} commits. Please pull changes.[/]")
                raise DirtyRepoError("Local branch is behind remote.")
        except Exception:
            pass
        # --------------------------------------

    def _execute_pipeline(self, task: Task, ws_path: Path, tools: "ToolExecutor",
                          failure_context: str, result: dict) -> tuple[dict, dict, dict, dict, list, bool, bool, bool, bool, dict]:
        """Run the dynamic pipeline. Returns (plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, pipeline_halted, result)."""
        plan: dict = {}
        impl: dict = {}
        rel: dict = {}
        sec: dict = {}
        applied: list[str] = []
        is_doc_only = False
        is_fast_mode = False
        test_ok = True
        is_maintenance = task.mode == "maintenance"
        is_evolution = task.mode == "evolution"
        pipeline_halted = False

        for step in self.config.pipeline:
            if pipeline_halted:
                break

            step_result = self._run_pipeline_step(
                step, task, ws_path, tools,
                failure_context=failure_context,
                plan=plan, impl=impl, rel=rel,
                is_doc_only=is_doc_only,
                is_fast_mode=is_fast_mode,
            )

            if step_result.payload.get("skipped"):
                # Debugger skip still needs state updates (test phase marking)
                if step.agent_role == "debugger":
                    self._state.test_passing = True
                    self._state.mark_phase("test")
                    self._state.persist(ws_path)
                continue

            # --- Per-step post-processing (preserves all existing behavior) ---

            if step.agent_role == "planner":
                if step_result.status == "error":
                    result["status"] = "plan_failed"
                    return plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, pipeline_halted, result
                plan = step_result.payload

                # Update TaskState with plan output
                self._state.plan_steps = [
                    StepState(
                        step_number=s.get("step_number", 0),
                        description=s.get("description", ""),
                        files=s.get("files", []),
                        action=s.get("action", ""),
                        do_not_touch=s.get("do_not_touch", []),
                        code_hint=s.get("code_hint", ""),
                    )
                    for s in plan.get("steps", [])
                ]
                self._state.files_in_scope = plan.get("files_likely_affected", [])
                self._state.estimated_complexity = plan.get(
                    "estimated_complexity", "medium"
                )
                self._state.requires_core_change = plan.get(
                    "requires_core_change", False
                )
                self._state.mark_phase("plan")
                self._state.persist(ws_path)

                # ── Boundary Validation (Plan-Level) ──
                try:
                    violations = self.boundary.check_plan(plan, self.allow_core)
                    if violations:
                        self._log_event("core_override", {"files": violations})
                        console.print(
                            f"[yellow]⚠ Core override granted for: {violations}[/]"
                        )
                except BoundaryViolation as e:
                    console.print(f"[red]🚫 {e}[/]")
                    result["status"] = "boundary_violation"
                    return plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, pipeline_halted, result

                # ── Governance Mode Routing ──
                if is_evolution:
                    self.config.intervention.pause_before_pr = True

                # Strict doc-only detection
                objective_lower = task.objective.lower()
                is_doc_only = (
                    is_maintenance
                    and any(
                        term in objective_lower
                        for term in ["doc", "documentation", "///"]
                    )
                    and task.risk_level == "low"
                    and all(
                        s.get("action") == "modify"
                        for s in plan.get("steps", [])
                    )
                    and any(
                        f.endswith(".rs")
                        for f in plan.get("files_likely_affected", [])
                    )
                )

            elif step.agent_role == "implementer":
                # ── Maintenance: Surgical Documentation Path ──
                if is_doc_only:
                    console.print(
                        "\n[bold dim]📄 [MAINTENANCE MODE] "
                        "Surgical documentation update — implementer bypassed.[/]"
                    )

                    impl = {
                        "changes": [],
                        "tests_added": [],
                        "commit_message": (
                            f"docs: update documentation for {task.task_id}"
                        ),
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
                        attest_controller_action(entry, self.run_id)

                # ── Standard Execution Path ──
                else:
                    if step_result.status == "error":
                        result["status"] = "implementation_failed"
                        return plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, pipeline_halted, result
                    impl = step_result.payload

                    # Update TaskState with implementation output
                    self._state.files_modified = [
                        c.get("file", "")
                        for c in impl.get("changes", [])
                        if c.get("action") in ("modify", "create")
                    ]
                    self._state.files_created = [
                        c.get("file", "")
                        for c in impl.get("changes", [])
                        if c.get("action") == "create"
                    ]
                    self._state.tests_added = [
                        t.get("file", "") for t in impl.get("tests_added", [])
                    ]
                    self._state.commit_message = impl.get("commit_message", "")
                    self._state.implementation_summary = impl.get("summary", "")
                    self._state.mark_phase("implement")
                    self._state.persist(ws_path)

                    is_high_complexity = plan.get(
                        "estimated_complexity", ""
                    ).lower() in ["high", "large", "unknown"]
                    if is_high_complexity:
                        console.print(
                            "  [dim]High complexity: allowing full-file rewrites.[/]"
                        )

                    try:
                        applied = apply_changes(
                            ws_path,
                            impl.get("changes", []),
                            boundary=self.boundary,
                            allow_core=self.allow_core,
                            allow_test_modifications=not is_maintenance,
                            allow_full_rewrite=True,
                        )
                        applied += apply_tests(
                            ws_path,
                            impl.get("tests_added", []),
                            allow_test_modifications=not is_maintenance,
                        )
                    except BoundaryViolation as e:
                        console.print(
                            f"[red]🚫 Boundary Violation during "
                            f"implementation: {e}[/]"
                        )
                        result["status"] = "boundary_violation"
                        return plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, pipeline_halted, result

                    for entry in applied:
                        console.print(f"  [cyan]{entry}[/]")
                        attest_controller_action(entry, self.run_id)

                # ── Patch & Surgical failure retry (one attempt) ──
                # Catch BOTH "FAIL" (surgical) and "PATCH_FAILED" (diffs)
                patch_failures = [
                    a for a in applied if "FAIL" in a or "PATCH_FAILED" in a
                ]

                if patch_failures:
                    console.print(
                        "[yellow]⚠ Edit failed to apply "
                        "(likely a whitespace mismatch). "
                        "Attempting one auto-repair...[/]"
                    )

                    for entry in applied:
                        console.print(f"  [cyan]{entry}[/]")
                        attest_controller_action(entry, self.run_id)

                    if any(
                        "FAIL" in a or "PATCH_FAILED" in a for a in applied
                    ):
                        console.print(
                            "[red]❌ Auto-repair failed. "
                            "Aborting to prevent corrupted PR.[/]"
                        )
                        result["status"] = "implementation_failed"
                        return plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, pipeline_halted, result

                    retry_result = self._retry_patch(
                        task, plan, ws_path, impl, applied
                    )

                    if retry_result.status == "error":
                        result["status"] = "implementation_failed"
                        return plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, pipeline_halted, result

                    applied = apply_changes(
                        ws_path,
                        retry_result.payload.get("changes", []),
                        boundary=self.boundary,
                        allow_core=self.allow_core,
                        allow_test_modifications=not is_maintenance,
                        allow_full_rewrite=True,
                    )

                    for entry in applied:
                        console.print(f"  [cyan]{entry}[/]")
                        attest_controller_action(entry, self.run_id)

                    if any(a.startswith("PATCH_FAILED") for a in applied):
                        console.print(
                            "[red]❌ Patch retry failed. Aborting.[/]"
                        )
                        result["status"] = "implementation_failed"
                        return plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, pipeline_halted, result

                # Compute fast_mode for downstream agents
                is_fast_mode = (
                    len(self._state.files_modified) <= 2
                    and self._state.estimated_complexity
                    in ("trivial", "small")
                )
                if is_fast_mode:
                    console.print(
                        "  [dim]Trivial change detected. "
                        "Forcing downstream agents into Fast Mode.[/]"
                    )

            elif step.agent_role == "debugger":
                test_ok = step_result.payload.get("test_passing", True)

                if not test_ok:
                    result["status"] = "tests_failed"
                    console.print(
                        "[red]❌ Fix loop exhausted. Tests still failing.[/]"
                    )
                    if not self._confirm("Continue to PR anyway?"):
                        return plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, pipeline_halted, result

                self._state.test_passing = test_ok
                self._state.mark_phase("test")
                self._state.persist(ws_path)

            elif step.agent_role == "testgen":
                pass

            elif step.agent_role == "security":
                sec = step_result.payload

                self._state.mark_phase("security")

                if sec.get("verdict") == "block":
                    self._state.security_verdict = "block"
                    console.print(
                        "[red]🚫 Security blocked this change.[/]"
                    )
                    print_security_issues(sec)

                    if self.auto_approve:
                        console.print(
                            "[red]❌ Auto-approve enabled. "
                            "Aborting dangerous PR.[/]"
                        )
                        result["status"] = "security_blocked"
                        return plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, pipeline_halted, result

                    if not self._confirm("Override security block?"):
                        result["status"] = "security_blocked"
                        return plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, pipeline_halted, result
                else:
                    # Normalize missing/empty verdict to "warn"
                    verdict = sec.get("verdict") or "warn"
                    self._state.security_verdict = verdict

                    if verdict == "warn":
                        console.print(
                            "[yellow]⚠ Security review returned warnings.[/]"
                        )
                        print_security_issues(sec)

                    # Prevent the generic error-halt from aborting
                    # the pipeline on a non-blocking verdict.
                    step_result.status = "success"

            elif step.agent_role == "release":
                rel = step_result.payload

                self._state.version_bump = rel.get("version_bump", "")
                self._state.changelog_entry = rel.get("changelog_entry", "")
                self._state.mark_phase("release")

            elif step.agent_role == "archivist":
                nova_result = step_result.payload

                if is_maintenance:
                    nova_result["should_write_adr"] = False

                if (
                    nova_result
                    and nova_result.get("should_write_adr")
                    and nova_result.get("adr")
                ):
                    adr_applied = write_adr(
                        ws_path, nova_result["adr"]
                    )
                    if adr_applied:
                        console.print(f"  [cyan]{adr_applied}[/]")
                        attest_controller_action(adr_applied, self.run_id)

                # Maintenance mode: forbid file create/delete
                # and out-of-scope edits
                if is_maintenance:
                    allowed_paths = set(
                        plan.get("files_likely_affected") or []
                    )
                    if not allowed_paths:
                        raise RuntimeError(
                            "Maintenance mode requires explicit "
                            "files_likely_affected"
                        )

                    for mpath in allowed_paths:
                        self._workspace._git(
                            "add", mpath, check=False
                        )

                    diff_output = self._workspace._git(
                        "diff",
                        "--cached",
                        "--name-status",
                        check=False,
                    )
                    lines = (
                        diff_output.splitlines() if diff_output else []
                    )

                    created, deleted, touched = [], [], []
                    for line in lines:
                        parts = line.split("\t", 1)
                        if len(parts) != 2:
                            continue
                        mstatus, mpath = parts
                        if mstatus == "A":
                            created.append(mpath)
                        elif mstatus == "D":
                            deleted.append(mpath)
                        else:
                            touched.append(mpath)

                    out_of_scope = [
                        p for p in touched if p not in allowed_paths
                    ]
                    if created or deleted or out_of_scope:
                        raise RuntimeError(
                            f"Maintenance violation. "
                            f"created={created} deleted={deleted} "
                            f"out_of_scope={out_of_scope}"
                        )

            # Generic halt for any required step that errors
            if step_result.status == "error" and step.required:
                result["status"] = f"{step.agent_role}_failed"
                pipeline_halted = True

        return plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, pipeline_halted, result

    def _finalize(self, task: Task, plan: dict, impl: dict, rel: dict, sec: dict,
                  ws_path: Path, is_doc_only: bool, is_fast_mode: bool, result: dict,
                  pipeline_halted: bool, tools: ToolExecutor) -> dict:
        """Commit changes, create PR, archive task. Returns updated result dict."""
        # ── Phase routing: doc-only defaults for downstream ──
        if is_doc_only:
            if not sec:
                sec = {"verdict": "pass", "issues": []}
            if not rel:
                rel = {
                    "version_bump": "none",
                    "reasoning": "Maintenance mode — documentation only",
                    "changelog_entry": "- Documentation updates",
                }

        if pipeline_halted:
            return result

        # ── 8. Commit + PR ──
        self._state.mark_phase("commit")
        self._state.persist(ws_path)

        commit_msg = impl.get("commit_message", f"glitchlab: {task.task_id}")
        self._workspace.commit(commit_msg)

        # --- NEW: Rebase Before PR ---
        if getattr(self.config, "automation", None) and getattr(self.config.automation, "rebase_before_pr", False):
            console.print("[dim]🔄 Rebasing onto origin/main to prevent conflicts...[/]")

            if not self._workspace.rebase(auto_abort=False):
                resolved = False
                if self.test_command:
                    console.print("[yellow]⚠️ Rebase conflict detected. Handing over to Debugger for auto-resolution...[/]")
                    # The fix loop will naturally detect the conflict markers as syntax errors/test failures!
                    resolved = self._run_fix_loop(task, ws_path, tools, impl)

                if not resolved:
                    # Agent couldn't fix it (or no tests available to verify the fix). Clean up.
                    self._workspace._worktree_git("rebase", "--abort", check=False)
                    result["status"] = "rebase_conflict"
                    console.print("[red]❌ Auto-resolution failed or no tests available. PR aborted.[/]")
                    return result
                else:
                    # Tests passed! The agent successfully removed the markers and fixed the logic.
                    self._workspace._worktree_git("add", "-A")

                    # Tell Git to use 'true' as the editor to automatically accept the rebase commit message
                    env = os.environ.copy()
                    env["GIT_EDITOR"] = "true"
                    subprocess.run(["git", "rebase", "--continue"], cwd=ws_path, env=env, check=False)
                    console.print("[bold green]✅ Rebase conflict auto-resolved by agent![/]")

        if getattr(self.config.intervention, "pause_before_pr", True):
            diff = self._workspace.diff_stat()
            console.print(Panel(diff, title="Diff Summary", border_style="cyan"))
            if not self._confirm("Create PR?"):
                result["status"] = "pr_cancelled"
                return result

        try:
            self._workspace.push(force=True)  # Force push required if we just rebased
            pr_url = self._create_pr(task, impl, rel)
            result["pr_url"] = pr_url
            result["status"] = "pr_created"
            console.print(f"[bold green]✅ PR created: {pr_url}[/]")

            # --- NEW: Auto-Merge ---
            if getattr(self.config, "automation", None) and getattr(self.config.automation, "auto_merge_pr", False):
                console.print("[dim]🚀 Auto-merge enabled. Squashing and merging...[/]")
                merge_res = subprocess.run(
                    ["gh", "pr", "merge", pr_url, "--squash"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )

                if merge_res.returncode == 0:
                    result["status"] = "merged"
                    console.print("[bold green]🎉 PR Auto-Merged successfully![/]")
                else:
                    # Graceful degradation: If CI/CD branch protections block the merge,
                    # it just stays open as a PR.
                    console.print(f"[yellow]⚠️ Auto-merge blocked by GitHub (likely awaiting CI). PR remains open.\n{merge_res.stderr.strip()}[/]")

        except Exception as e:
            logger.warning(f"PR creation failed: {e}")
            result["status"] = "committed"
            result["branch"] = self._workspace.branch_name
            console.print(f"[yellow]Changes committed to {self._workspace.branch_name}[/]")

        print_budget_summary(self.router.budget.summary())
        result["events"] = self._state.events
        result["budget"] = self.router.budget.summary()

        if getattr(task, "file_path", None) and task.file_path.exists() and task.file_path.parent.name == "queue":
            archive_dir = task.file_path.parent.with_name("archive")
            archive_dir.mkdir(parents=True, exist_ok=True)
            task.file_path.rename(archive_dir / task.file_path.name)
            console.print(f"[dim]Moved task file to {archive_dir / task.file_path.name}[/]")

        return result

    def _startup(self, task: Task) -> tuple[Path, "ToolExecutor", str]:
        """Create workspace, build indexes, load constraints. Returns (ws_path, tools, failure_context)."""
        # ── 1. Create workspace ──
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

        # ── 1.5. Build repo index (file map for planner) ──
        console.print("\n[bold dim]🗂  [INDEX] Scanning repository...[/]")
        self._repo_index = build_index(ws_path)
        self._repo_index_context = self._repo_index.to_agent_context(max_files=200)
        console.print(
            f"  [dim]{self._repo_index.total_files} files, "
            f"{len(self._repo_index.languages)} languages[/]"
        )
        self._log_event("repo_indexed", {
            "total_files": self._repo_index.total_files,
            "languages": self._repo_index.languages,
        })

        # ── 1.6. Initialize ScopeResolver (Layer 1) ──
        self._scope = ScopeResolver(ws_path, self._repo_index)

        # ── 1.7. Prelude: load constraints only (not global prefix) ──
        if self._prelude.available:
            console.print("[bold dim]📋 [PRELUDE] Loading constraints...[/]")
            self._prelude.refresh()
            prelude_constraints = self._prelude.get_constraints()
            if prelude_constraints:
                task.constraints = list(set(task.constraints + prelude_constraints))
                console.print(f"  [dim]{len(prelude_constraints)} constraints merged[/]")
            self._log_event("prelude_constraints_loaded", {
                "count": len(prelude_constraints) if prelude_constraints else 0,
            })

        # ── 1.8. Load failure context from history ──
        failure_context = self._history.build_failure_context()
        if failure_context:
            console.print("  [dim]Loaded recent failure patterns for planner[/]")

        return ws_path, tools, failure_context

    def run(self, task: Task) -> dict[str, Any]:
        """Execute the full agent pipeline for a task."""

        # --- NEW: Generate Session Identity for Zephyr ---
        self.run_id = str(uuid.uuid4())
        bus.emit(
            event_type="run.started",
            payload={"task_id": task.task_id, "objective": task.objective},
            run_id=self.run_id
        )

        # Ensure we plan against the most recent code.
        # Soft-fail to avoid breaking offline/CI scenarios.
        pre_task_git_fetch(self.repo_path)

        self._check_repo_clean()

        # Initialize structured task state
        self._state = TaskState(
            task_id=task.task_id,
            objective=task.objective,
            mode=task.mode or "evolution",
            risk_level=task.risk_level,
        )

        result: dict[str, Any] = {
            "task_id": task.task_id,
            "status": "pending",
            "events": [],
        }

        self._print_banner(task)

        try:
            ws_path, tools, failure_context = self._startup(task)

            plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, pipeline_halted, result = self._execute_pipeline(
                task, ws_path, tools, failure_context, result
            )

            result = self._finalize(task, plan, impl, rel, sec, ws_path, is_doc_only, is_fast_mode, result, pipeline_halted, tools)

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
            self._history.record(result)
            self._write_session_entry(task, result)

        # --- NEW: Zephyr Quality Scoring & Run Completion ---
        quality_score = calculate_quality_score(
            self.router.budget.summary(), self._state
        )
        result["quality_score"] = quality_score

        bus.emit(
            event_type="run.completed",
            payload=result,
            run_id=self.run_id,
            metadata={"quality_score": quality_score}
        )

        return result

    # -----------------------------------------------------------------------
    # Prelude Session Entry
    # -----------------------------------------------------------------------

    def _write_session_entry(self, task: Task, result: dict[str, Any]) -> None:
        """Write a Prelude-compatible session entry to .context/session.json."""
        session_file = self.repo_path / ".context" / "session.json"
        if not session_file.parent.exists():
            return  # No .context/ directory — Prelude not initialized

        # Map GLITCHLAB status to Prelude outcome
        status = result.get("status", "unknown")
        outcome_map = {
            "pr_created": "success",
            "committed": "success",
            "merged": "success",
            "implementation_failed": "failed",
            "budget_exceeded": "failed",
            "error": "failed",
            "interrupted": "failed",
        }
        outcome = outcome_map.get(status, "partial")

        # Build the entry
        entry = {
            "id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "prompt",
            "summary": task.objective[:200],
            "filesAffected": list(set(
                (self._state.files_modified if self._state else [])
                + (self._state.files_created if self._state else [])
            )),
            "outcome": outcome,
            "tags": [self._state.mode if self._state else "unknown"],
        }

        # Read existing, append, cap at 20 entries, write back
        try:
            if session_file.exists():
                data = json.loads(session_file.read_text())
            else:
                data = {
                    "$schema": "https://adjective.us/prelude/schemas/v1/session.json",
                    "version": "1.0.0",
                    "sessions": [{
                        "sessionId": "glitchlab",
                        "startedAt": datetime.now(timezone.utc).isoformat(),
                        "entries": []
                    }]
                }

            entries = data.get("sessions", [{}])[0].get("entries", [])
            entries.append(entry)

            # Cap at 20 entries
            if len(entries) > 20:
                entries = entries[-20:]

            data["sessions"][0]["entries"] = entries
            session_file.write_text(json.dumps(data, indent=2))
            logger.debug(
                f"[PRELUDE] Session entry written: {outcome} — {task.objective[:60]}"
            )
        except Exception as e:
            logger.warning(f"[PRELUDE] Failed to write session entry: {e}")

    # -----------------------------------------------------------------------
    # Pipeline Step Dispatcher
    # -----------------------------------------------------------------------

    def _run_pipeline_step(
        self,
        step: PipelineStep,
        task: Task,
        ws_path: Path,
        tools: ToolExecutor,
        *,
        failure_context: str = "",
        plan: dict | None = None,
        impl: dict | None = None,
        rel: dict | None = None,
        is_doc_only: bool = False,
        is_fast_mode: bool = False,
    ) -> AgentResult:
        """Execute a single pipeline step by dispatching to the registered agent."""

        bus.emit(
            event_type="pipeline.step_started",
            payload={
                "step_name": step.name,
                "agent_role": step.agent_role,
                "required": step.required,
                "skip_if": step.skip_if,
            },
            agent_id=step.agent_role,
            run_id=getattr(self, 'run_id', None),
        )

        # 1. Check skip_if conditions against current state
        skip_conditions: dict[str, bool] = {
            "doc_only": is_doc_only,
            "fast_mode": is_fast_mode,
            "no_test_command": not self.test_command,
        }
        for condition in step.skip_if:
            if skip_conditions.get(condition, False):
                bus.emit(
                    event_type="pipeline.step_skipped",
                    payload={
                        "step_name": step.name,
                        "agent_role": step.agent_role,
                        "skip_reason": condition,
                    },
                    agent_id=step.agent_role,
                    run_id=getattr(self, 'run_id', None),
                )
                return AgentResult(
                    status="success",
                    agent=step.agent_role,
                    payload={"skipped": True},
                )

        # 2. Dispatch to the appropriate _run_* method based on agent_role
        role = step.agent_role

        if role == "planner":
            result = self._run_planner(task, ws_path, failure_context)
            bus.emit(
                event_type="pipeline.step_completed",
                payload={"step_name": step.name, "agent_role": role, "status": result.status},
                agent_id=role, run_id=getattr(self, 'run_id', None),
            )
            return result

        if role == "implementer":
            if is_doc_only:
                # Doc-only path: implementer bypassed, handled in post-processing
                result = AgentResult(
                    status="success", agent="implementer",
                    payload={"doc_only": True},
                )
                bus.emit(
                    event_type="pipeline.step_completed",
                    payload={
                        "step_name": step.name, "agent_role": role, "status": result.status,
                    },
                    agent_id=role, run_id=getattr(self, 'run_id', None),
                )
                return result
            result = self._run_implementer(task, plan or {}, ws_path, tools)
            bus.emit(
                event_type="pipeline.step_completed",
                payload={"step_name": step.name, "agent_role": role, "status": result.status},
                agent_id=role, run_id=getattr(self, 'run_id', None),
            )
            return result

        if role == "testgen":
            self._run_testgen(task, ws_path, is_doc_only)
            result = AgentResult(status="success", agent="testgen")
            bus.emit(
                event_type="pipeline.step_completed",
                payload={"step_name": step.name, "agent_role": role, "status": result.status},
                agent_id=role, run_id=getattr(self, 'run_id', None),
            )
            return result

        if role == "debugger":
            test_ok = self._run_fix_loop(task, ws_path, tools, impl or {})
            result = AgentResult(
                status="success",
                agent="debugger",
                payload={"test_passing": test_ok},
            )
            bus.emit(
                event_type="pipeline.step_completed",
                payload={"step_name": step.name, "agent_role": role, "status": result.status},
                agent_id=role, run_id=getattr(self, 'run_id', None),
            )
            return result

        if role == "security":
            result = self._run_security(task, impl or {}, ws_path, is_fast_mode)
            bus.emit(
                event_type="pipeline.step_completed",
                payload={"step_name": step.name, "agent_role": role, "status": result.status},
                agent_id=role, run_id=getattr(self, 'run_id', None),
            )
            return result

        if role == "release":
            result = self._run_release(task, impl or {}, ws_path, is_fast_mode)
            bus.emit(
                event_type="pipeline.step_completed",
                payload={"step_name": step.name, "agent_role": role, "status": result.status},
                agent_id=role, run_id=getattr(self, 'run_id', None),
            )
            return result

        if role == "archivist":
            result = self._run_archivist(
                task, impl or {}, plan or {}, rel or {}, ws_path, is_fast_mode
            )
            bus.emit(
                event_type="pipeline.step_completed",
                payload={"step_name": step.name, "agent_role": role, "status": result.status},
                agent_id=role, run_id=getattr(self, 'run_id', None),
            )
            return result

        raise ValueError(f"Unknown pipeline agent_role: {role!r}")

    # -----------------------------------------------------------------------
    # Agent Runners — v2: Surgical context via TaskState + ScopeResolver
    # -----------------------------------------------------------------------

    def _run_planner(self, task: Task, ws_path: Path, failure_context: str = "") -> AgentResult:
        console.print("\n[bold magenta]🧠 [ZAP] Planning...[/]")

        # Planner gets: repo file map + task + failure history
        # NO global Prelude dump. Prelude constraints already merged into task.
        objective_parts = []

        if self._repo_index_context:
            objective_parts.append(self._repo_index_context)

        if failure_context:
            objective_parts.append(failure_context)

        objective_parts.append(f"TASK:\n{task.objective}")

        objective = "\n\n---\n\n".join(objective_parts)

        symbol_index = SymbolIndex(ws_path)

        context = AgentContext(
            task_id=task.task_id,
            run_id=self.run_id,
            objective=objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            constraints=task.constraints,
            acceptance_criteria=task.acceptance_criteria,
            risk_level=task.risk_level,
            extra={
                "prelude": self._prelude,
                "symbol_index": symbol_index,
            },
        )

        raw = self.agents["planner"].run(context)
        self._log_event("plan_created", {
            "steps": len(raw.get("steps", [])),
            "risk": raw.get("risk_level"),
        })

        print_plan(raw)

        if self.config.intervention.pause_after_plan and not self.auto_approve:
            if not self._confirm("Approve plan?"):
                raw["_aborted"] = True
                raw["parse_error"] = True
                return AgentResult.from_raw(raw)

        return AgentResult.from_raw(raw)

    def _run_implementer(self, task: Task, plan: dict, ws_path: Path, tools: ToolExecutor) -> AgentResult:
        console.print("\n[bold blue]🔧 [PATCH] Implementing...[/]")

        # --- AST LAYER INITIALIZATION ---
        symbol_index = SymbolIndex(ws_path)

        # KEEP THIS: ScopeResolver computes context from actual imports
        # Gives the tool-loop a great starting point so it doesn't have to read_file blindly
        file_context = self._scope.resolve_for_files(
            plan.get("files_likely_affected", []),
            include_deps=True,
            signatures_only=True,
        )

        # Keep the user's task constraints, but DROP the JSON formatting constraints
        impl_constraints = list(task.constraints)

        # Add plan-level do_not_touch as constraints
        plan_dnt = plan.get("do_not_touch", [])
        if plan_dnt:
            impl_constraints.append(f"DO NOT TOUCH these files/symbols: {', '.join(plan_dnt)}")

        # --- MEMORY INJECTION ---
        heuristics = self._history.build_heuristics(plan.get("files_likely_affected", []))

        # Pass structured task state AND the tool executor
        context = AgentContext(
            task_id=task.task_id,
            run_id=self.run_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            constraints=impl_constraints,
            acceptance_criteria=task.acceptance_criteria,
            file_context=file_context,
            previous_output=self._state.to_agent_summary("implementer"),
            extra={
                "tool_executor": tools, # <-- WIRE THE KEYS TO THE SANDBOX HERE
                "test_command": self.test_command, # Optional: let Patch run the primary test
                "learned_heuristics": heuristics,
                "symbol_index": symbol_index,
                "prelude": self._prelude,
                "fast_mode": (
                    len(self._state.files_in_scope) <= 3
                    and self._state.estimated_complexity in ("trivial", "small")
                ),
            },
        )

        # --- THE SWITCHBOARD DELEGATION LOOP ---
        while True:
            impl = self.agents["implementer"].run(context, max_tokens=12000)
            
            # Did Patch yield to ask for help?
            if impl.get("_status") == "delegating":
                target = impl.get("colleague", "unknown")
                request = impl.get("request", "No specific request provided.")
                tc_id = impl.get("tc_id")
                tc_name = impl.get("tc_name")
                
                console.print(f"\n[bold magenta]📞 Patch is tagging in {target.upper()}...[/]")
                console.print(f"  [dim]Request: {request}[/]")
                
                # 1. Spin up the requested agent
                colleague_response = self._run_delegated_agent(target, request, task, ws_path, tools)
                
                # 2. Inject the response back into Patch's memory so it can resume seamlessly
                context.extra["_resume_messages"] = impl.get("_messages", [])
                context.extra["_resume_messages"].append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tc_name,
                    "content": f"Colleague {target.upper()} responded:\n{colleague_response}"
                })
                
                console.print("[bold blue]🔄 Resuming Patch...[/]")
                continue
                
            break  # Exit loop when Patch successfully calls `done` (or hits a hard error)

        # --- MEMORY EXTRACTION ---
        # Capture _messages from raw dict before wrapping (from_raw strips _ keys)
        messages = impl.get("_messages", [])
        impl_result = AgentResult.from_raw(impl)
        if messages:
            outcome = "fail" if impl_result.status == "error" else "pass"
            patterns = extract_patterns_from_messages(messages, outcome)
            if patterns:
                self._history.record_patterns(task.task_id, patterns)

        # For doc-comment tasks, use surgical insertion
        for change in impl_result.payload.get("changes", []):
            if change.get("action") == "modify" and not change.get("_already_applied"):
                fpath = ws_path / change["file"]
                if fpath.exists():
                    inserted = insert_doc_comments(fpath, self.router)
                    if inserted:
                        change["_doc_inserted"] = True
                        change["content"] = None

        self._log_event("implementation_created", {
            "changes": len(impl_result.payload.get("changes", [])),
            "tests": len(impl_result.payload.get("tests_added", [])),
        })

        return impl_result

    def _run_delegated_agent(self, target: str, request: str, task: Task, ws_path: Path, tools: ToolExecutor) -> str:
        return run_delegated_agent(
            target=target, request=request, task=task, ws_path=ws_path,
            run_id=self.run_id, repo_path=self.repo_path, agents=self.agents,
            tools=tools, prelude=self._prelude, repo_index=self._repo_index,
            test_command=self.test_command,
        )

    def _retry_patch(
        self,
        task: Task,
        plan: dict,
        ws_path: Path,
        original_impl: dict,
        applied_entries: list[str],
    ) -> AgentResult:
        console.print("[dim]Re-prompting implementer with git error context...[/]")

        error_lines = [
            a.replace("PATCH_ERROR ", "")
            for a in applied_entries
            if a.startswith("PATCH_ERROR")
        ]
        git_error = "\n".join(error_lines)

        file_context = self._scope.resolve_for_files(
            plan.get("files_likely_affected", []),
            include_deps=False,
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
            run_id=self.run_id,
            objective=task.objective + "\n\n" + retry_prompt,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            constraints=task.constraints,
            acceptance_criteria=task.acceptance_criteria,
            file_context=file_context,
            previous_output=self._state.to_agent_summary("implementer"),
        )

        raw = self.agents["implementer"].run(context, max_tokens=8192)
        return AgentResult.from_raw(raw)
    
    def _run_testgen(self, task: Task, ws_path: Path, is_doc_only: bool) -> None:
        """Run the Shield agent to generate a regression test if none exists."""
        if is_doc_only:
            return

        # Check if tests already created by the implementer
        existing_tests = self._state.tests_added[:]
        for f in self._state.files_created + self._state.files_modified:
            if "test_" in f.lower() or "_test" in f.lower() or f.startswith("tests/"):
                existing_tests.append(f)

        if existing_tests:
            console.print(f"  [dim]Tests already exist/created: {existing_tests[0]}. Skipping Shield.[/]")
            return

        console.print("\n[bold green]🛡️ [SHIELD] Generating regression test...[/]")
        
        # Resolve actual written code for Shield to analyze
        file_context = self._scope.resolve_for_files(
            self._state.files_modified,
            include_deps=False,
        )
        
        context = AgentContext(
            task_id=task.task_id,
            run_id=self.run_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            previous_output=self._state.to_agent_summary("testgen"),
            file_context=file_context,
            extra={"test_command": self.test_command}
        )
        
        raw = self.agents["testgen"].run(context)
        tg_result = AgentResult.from_raw(raw)

        if tg_result.status == "error" or not tg_result.payload.get("test_file"):
            console.print("  [yellow]Shield failed to generate a valid test. Continuing.[/]")
            return

        test_file = tg_result.payload["test_file"]
        content = tg_result.payload["content"]
        desc = tg_result.payload["description"]
        
        try:
            fpath = ws_path / test_file
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)
            self._state.tests_added.append(test_file)
            self._log_event("testgen_created", {"file": test_file, "description": desc})
            console.print(f"  [cyan]TESTGEN {test_file}[/]")
            console.print(f"  [dim]Generated: {desc}[/]")
            attest_controller_action(f"TESTGEN {test_file}", self.run_id)
        except Exception as e:
            console.print(f"  [red]Failed to write test file: {e}[/]")

    def _run_fix_loop(
        self, task: Task, ws_path: Path, tools: ToolExecutor, impl: dict
    ) -> bool:
        """
        Run test → debug → fix loop (v3.0).
        Debugger is now agentic and manages its own tool-loop to investigate and fix.
        """
        max_attempts = self.config.limits.max_fix_attempts

        for attempt in range(1, max_attempts + 1):
            console.print(f"\n[bold]🧪 Test run {attempt}/{max_attempts}...[/]")

            try:
                # 1. Execute test command to see if a fix is even needed
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

            # 2. Invoke Debugger (Agentic Loop)
            console.print(f"\n[bold yellow]🐛 [REROUTE] Debugging (attempt {attempt})...[/]")

            context = AgentContext(
                task_id=task.task_id,
                run_id=self.run_id,
                objective=task.objective,
                repo_path=str(self.repo_path),
                working_dir=str(ws_path),
                previous_output=self._state.to_agent_summary("debugger"),
                extra={
                    "error_output": (error_output or "")[-3000:],
                    "test_command": self.test_command,
                    "tool_executor": tools, # Hand over the keys to the sandbox
                    "prelude": self._prelude,
                    "repo_index": self._repo_index,
                    "fast_mode": (
                        len(self._state.files_in_scope) <= 3
                        and self._state.estimated_complexity in ("trivial", "small")
                    ),
                },
            )

            # Debugger now runs its own 10-step loop internally
            raw_debug = self.agents["debugger"].run(context)
            debug_result = AgentResult.from_raw(raw_debug)

            # Record debug Turn for TaskHistory
            self._state.previous_fixes.append(debug_result.payload)
            self._state.last_error = debug_result.payload.get("diagnosis", "Unknown error")
            self._state.debug_attempts = attempt

            # --- RECORD FAILURE MEMORY ---
            fix_changes = debug_result.payload.get("fix", {}).get("changes", [])
            for change in fix_changes:
                if change.get("file"):
                    self._history.record_failure_detail(
                        task_id=task.task_id,
                        file_modified=change["file"],
                        error_type=self._state.last_error,
                        resolution=debug_result.payload.get(
                            "root_cause", "Fixed in debug loop"
                        ),
                    )

            # Sync TaskState with files written by the debugger's tools
            fix_changes = debug_result.payload.get("fix", {}).get("changes", [])
            for change in fix_changes:
                f = change.get("file")
                if f:
                    if change.get("action") == "create" and f not in self._state.files_created:
                        self._state.files_created.append(f)
                    elif f not in self._state.files_modified:
                        self._state.files_modified.append(f)

            if debug_result.status == "error":
                console.print("[yellow]⚠ Debugger failed to conclude. Retrying loop...[/]")
                continue

            if not debug_result.payload.get("should_retry", False):
                console.print("[yellow]Debugger suggests abandoning fix.[/]")
                break

        return False

    def _run_auditor(self, task: Task, impl: dict, ws_path: Path) -> dict:
        result = run_auditor(
            agent=self.auditor, task=task, ws_path=ws_path, run_id=self.run_id,
            repo_path=self.repo_path, state=self._state, workspace=self._workspace,
        )
        self._log_event("auditor_feedback", {"feedback": result.get("feedback")})
        return result

    def _run_security(self, task: Task, impl: dict, ws_path: Path, is_fast_mode: bool = False) -> AgentResult:
        result = run_security(
            agent=self.agents["security"], task=task, ws_path=ws_path,
            run_id=self.run_id, repo_path=self.repo_path, state=self._state,
            workspace=self._workspace, config=self.config, repo_index=self._repo_index,
            prelude=self._prelude, is_fast_mode=is_fast_mode,
        )
        self._log_event("security_review", {"verdict": result.payload.get("verdict")})
        return result

    def _run_release(self, task: Task, impl: dict, ws_path: Path, is_fast_mode: bool = False) -> AgentResult:
        result = run_release(
            agent=self.agents["release"], task=task, ws_path=ws_path,
            run_id=self.run_id, repo_path=self.repo_path, state=self._state,
            workspace=self._workspace, is_fast_mode=is_fast_mode,
        )
        self._log_event("release_assessment", {"bump": result.payload.get("version_bump")})
        return result

    def _run_archivist(
        self, task: Task, impl: dict, plan: dict, release: dict, ws_path: Path, is_fast_mode: bool = False
    ) -> AgentResult:
        return run_archivist(
            agent=self.agents["archivist"], task=task, ws_path=ws_path,
            run_id=self.run_id, repo_path=self.repo_path, state=self._state,
            is_fast_mode=is_fast_mode,
        )

    # -----------------------------------------------------------------------
    # PR Creation
    # -----------------------------------------------------------------------

    def _create_pr(self, task: Task, impl: dict, release: dict) -> str:
        """Create a GitHub PR via gh CLI."""
        title = impl.get("commit_message", f"glitchlab: {task.task_id}")
        body = build_pr_body(task, impl, release)

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
            "task_id": self._state.task_id if self._state else None,
            "data": data or {},
        }
        if self._state:
            self._state.events.append(event)
        logger.debug(f"[EVENT] {event_type}: {data}")