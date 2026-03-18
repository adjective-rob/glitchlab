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

import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Literal, Any
from pydantic import BaseModel, Field

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table

from glitchlab.agents import AgentContext, BaseAgent, AgentResult
from glitchlab.registry import AGENT_REGISTRY, get_agent
from glitchlab.config_loader import GlitchLabConfig, PipelineStep, load_config
from glitchlab.doc_inserter import insert_doc_comments
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

console = Console()


# ---------------------------------------------------------------------------
# Task State — Structured Working Memory (Layer 3)
# ---------------------------------------------------------------------------

class StepState(BaseModel):
    """Tracks the status of an individual planned step."""
    step_number: int
    description: str = ""
    files: list[str] = Field(default_factory=list)
    action: str = ""
    status: Literal["pending", "completed", "failed", "skipped"] = "pending"
    outcome: str = ""


class TaskState(BaseModel):
    """
    Structured working memory that flows between agents.

    Replaces the old pattern of passing raw `previous_output` blobs.
    Each agent reads what it needs and writes its contribution.
    The Controller owns this object and persists it per-run.
    """

    task_id: str
    objective: str
    mode: str = "evolution"
    risk_level: str = "low"

    # Planner output (consumed by Implementer, Debugger, Security)
    plan_steps: list[StepState] = Field(default_factory=list)
    files_in_scope: list[str] = Field(default_factory=list)
    estimated_complexity: str = "medium"
    requires_core_change: bool = False

    # Implementer output (consumed by Debugger, Security, Release)
    files_modified: list[str] = Field(default_factory=list)
    files_created: list[str] = Field(default_factory=list)
    tests_added: list[str] = Field(default_factory=list)
    commit_message: str = ""
    implementation_summary: str = ""

    # Debug loop state
    test_passing: bool = False
    debug_attempts: int = 0
    last_error: str = ""
    previous_fixes: list[dict] = Field(default_factory=list)

    # Security + Release
    security_verdict: str = ""
    version_bump: str = ""
    changelog_entry: str = ""

    # Tracking
    completed_phases: list[str] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)

    AGENT_FIELDS: ClassVar[dict[str, list[str]]] = {
        "planner": ["previous_fixes"],
        "implementer": ["plan_steps", "files_in_scope", "estimated_complexity"],
        "testgen": ["files_modified", "files_created", "implementation_summary"],
        "debugger": ["files_modified", "files_created", "last_error",
                     "debug_attempts", "previous_fixes"],
        "auditor": ["files_modified", "files_created", "implementation_summary"],
        "security": ["files_modified", "files_created", "implementation_summary"],
        "release": ["files_modified", "implementation_summary", "security_verdict"],
        "archivist": ["plan_steps", "files_modified", "implementation_summary",
                      "version_bump"],
    }

    FIELD_CAPS: ClassVar[dict[tuple[str, str], int]] = {
        ("planner", "previous_fixes"): 3,
        ("debugger", "previous_fixes"): 2,
    }

    def mark_phase(self, phase: str) -> None:
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)

    def to_agent_summary(self, for_agent: str) -> dict:
        """
        Return only the fields relevant to a specific agent.
        This is the core of the context-router pattern: agents get
        precisely what they need, not everything.

        Field routing is driven by AGENT_FIELDS (which fields each agent
        sees) and FIELD_CAPS (optional tail-slice limits for list fields).
        New agent roles can be added by extending AGENT_FIELDS.
        """
        base = {
            "task_id": self.task_id,
            "objective": self.objective,
            "mode": self.mode,
            "risk_level": self.risk_level,
        }
        fields = self.AGENT_FIELDS.get(for_agent, [])
        for field_name in fields:
            value = getattr(self, field_name, None)
            cap = self.FIELD_CAPS.get((for_agent, field_name))
            if cap is not None:
                value = value[-cap:] if value else []
            elif isinstance(value, list) and all(
                hasattr(v, "model_dump") for v in value
            ):
                value = [v.model_dump() for v in value]
            base[field_name] = value
        return base

    def persist(self, ws_path: Path) -> None:
        """Write current state to workspace for debugging/auditing."""
        state_dir = ws_path / ".glitchlab"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "task_state.json").write_text(
            self.model_dump_json(indent=2)
        )

class DirtyRepoError(Exception):
    """Raised when the main repository has uncommitted changes."""
    pass


# ---------------------------------------------------------------------------


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

    # -----------------------------------------------------------------------
    # Git sync (pre-task)
    # -----------------------------------------------------------------------

    def _is_git_repo(self, path: Path) -> bool:
        """Return True if `path` looks like a git working tree."""
        try:
            if (path / ".git").exists():
                return True
            # Worktrees can have a .git file pointing to the actual gitdir
            if (path / ".git").is_file():
                return True
        except Exception:
            return False
        return False

    def _run_git(self, args: list[str], cwd: Path, timeout: int = 20) -> subprocess.CompletedProcess:
        """Run a git command and capture output for logging."""
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _pre_task_git_fetch(self) -> None:
        """Best-effort fetch to ensure planning is against recent `origin/main`.

        Soft-fails (warn + continue) to avoid breaking offline/CI runs.
        """
        if not self._is_git_repo(self.repo_path):
            logger.debug(f"[GIT] Skipping fetch: not a git repo: {self.repo_path}")
            return

        try:
            res = self._run_git(["fetch", "origin", "main"], cwd=self.repo_path, timeout=20)
            if res.returncode != 0:
                stderr = (res.stderr or "").strip()
                stdout = (res.stdout or "").strip()
                msg = stderr or stdout or f"git fetch failed with code {res.returncode}"
                logger.warning(f"[GIT] Pre-task fetch failed (soft): {msg}")
                return

            out = (res.stdout or "").strip()
            if out:
                logger.info(f"[GIT] Pre-task fetch: {out}")
            else:
                logger.debug("[GIT] Pre-task fetch: up to date")
        except Exception as e:
            logger.warning(f"[GIT] Pre-task fetch exception (soft): {e}")
            return

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
        self._pre_task_git_fetch()

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

        console.print(Panel(
            f"[bold green]Task:[/] {task.objective[:120]}\n"
            f"[bold]ID:[/] {task.task_id}  |  [bold]Source:[/] {task.source}\n"
            f"[bold]Risk:[/] {task.risk_level}  |  [bold]Mode:[/] {task.mode.upper()}",
            title="⚡ GLITCHLAB v2",
            subtitle="Build Weird. Ship Clean.",
            border_style="bright_green",
        ))

        try:
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

            # ── 2. Dynamic Pipeline ──
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
                        return result
                    plan = step_result.payload

                    # Update TaskState with plan output
                    self._state.plan_steps = [
                        StepState(
                            step_number=s.get("step_number", 0),
                            description=s.get("description", ""),
                            files=s.get("files", []),
                            action=s.get("action", ""),
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
                        return result

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
                            self._attest_controller_action(entry)

                    # ── Standard Execution Path ──
                    else:
                        if step_result.status == "error":
                            result["status"] = "implementation_failed"
                            return result
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
                        ).lower() in ["high", "large"]
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
                            return result

                        for entry in applied:
                            console.print(f"  [cyan]{entry}[/]")
                            self._attest_controller_action(entry)

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
                            self._attest_controller_action(entry)

                        if any(
                            "FAIL" in a or "PATCH_FAILED" in a for a in applied
                        ):
                            console.print(
                                "[red]❌ Auto-repair failed. "
                                "Aborting to prevent corrupted PR.[/]"
                            )
                            result["status"] = "implementation_failed"
                            return result

                        retry_result = self._retry_patch(
                            task, plan, ws_path, impl, applied
                        )

                        if retry_result.status == "error":
                            result["status"] = "implementation_failed"
                            return result

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
                            self._attest_controller_action(entry)

                        if any(a.startswith("PATCH_FAILED") for a in applied):
                            console.print(
                                "[red]❌ Patch retry failed. Aborting.[/]"
                            )
                            result["status"] = "implementation_failed"
                            return result

                elif step.agent_role == "debugger":
                    test_ok = step_result.payload.get("test_passing", True)

                    if not test_ok:
                        result["status"] = "tests_failed"
                        console.print(
                            "[red]❌ Fix loop exhausted. Tests still failing.[/]"
                        )
                        if not self._confirm("Continue to PR anyway?"):
                            return result

                    self._state.test_passing = test_ok
                    self._state.mark_phase("test")
                    self._state.persist(ws_path)

                elif step.agent_role == "testgen":
                    pass

                elif step.agent_role == "security":
                    sec = step_result.payload

                    # --- FAST MODE CHECK ---
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

                    self._state.security_verdict = sec.get("verdict", "")
                    self._state.mark_phase("security")

                    if sec.get("verdict") == "block":
                        console.print(
                            "[red]🚫 Security blocked this change.[/]"
                        )
                        self._print_security_issues(sec)

                        if self.auto_approve:
                            console.print(
                                "[red]❌ Auto-approve enabled. "
                                "Aborting dangerous PR.[/]"
                            )
                            result["status"] = "security_blocked"
                            return result

                        if not self._confirm("Override security block?"):
                            result["status"] = "security_blocked"
                            return result

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
                        adr_applied = self._write_adr(
                            ws_path, nova_result["adr"]
                        )
                        if adr_applied:
                            console.print(f"  [cyan]{adr_applied}[/]")
                            self._attest_controller_action(adr_applied)

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
                    console.print(f"[dim]🚀 Auto-merge enabled. Squashing and merging...[/]")
                    merge_res = subprocess.run(
                        ["gh", "pr", "merge", pr_url, "--squash"],
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True
                    )
                    
                    if merge_res.returncode == 0:
                        result["status"] = "merged"
                        console.print(f"[bold green]🎉 PR Auto-Merged successfully![/]")
                    else:
                        # Graceful degradation: If CI/CD branch protections block the merge, 
                        # it just stays open as a PR.
                        console.print(f"[yellow]⚠️ Auto-merge blocked by GitHub (likely awaiting CI). PR remains open.\n{merge_res.stderr.strip()}[/]")

            except Exception as e:
                logger.warning(f"PR creation failed: {e}")
                result["status"] = "committed"
                result["branch"] = self._workspace.branch_name
                console.print(f"[yellow]Changes committed to {self._workspace.branch_name}[/]")

            self._print_budget_summary()
            result["events"] = self._state.events
            result["budget"] = self.router.budget.summary()

            if getattr(task, "file_path", None) and task.file_path.exists() and task.file_path.parent.name == "queue":
                archive_dir = task.file_path.parent.with_name("archive")
                archive_dir.mkdir(parents=True, exist_ok=True)
                task.file_path.rename(archive_dir / task.file_path.name)
                console.print(f"[dim]Moved task file to {archive_dir / task.file_path.name}[/]")

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

        # --- NEW: Zephyr Quality Scoring & Run Completion ---
        quality_score = self._calculate_quality_score()
        result["quality_score"] = quality_score

        bus.emit(
            event_type="run.completed",
            payload=result,
            run_id=self.run_id,
            metadata={"quality_score": quality_score}
        )

        return result

    def _calculate_quality_score(self) -> dict[str, Any]:
        """Calculate a run quality score out of 100 based on efficiency and convergence."""
        score = 100
        budget_summary = self.router.budget.summary()
        
        # 1. Time & Efficiency (Penalize excessive token usage)
        total_tokens = budget_summary.get("total_tokens", 0)
        if total_tokens > 50000:
            score -= min(30, (total_tokens - 50000) // 5000) # Max penalty 30
        
        # 2. Convergence (Did it struggle in the fix loop?)
        debug_attempts = 0
        if self._state:
            debug_attempts = self._state.debug_attempts
            if debug_attempts > 0:
                score -= (debug_attempts * 10) # Heavy penalty for needing multiple fix attempts

        return {
            "score": max(0, score),
            "tokens_used": total_tokens,
            "debug_attempts": debug_attempts
        }
    
    def _attest_controller_action(self, action_summary: str) -> None:
        """Emit an SBOF attestation for direct controller file writes."""
        if action_summary.startswith("FAIL") or "ERROR" in action_summary:
            return
            
        bus.emit(
            event_type="action.completed",
            payload={
                "command": "controller.write_file",
                "stdout": action_summary,
                "stderr": "",
                "returncode": 0,
                "allowed": True
            },
            agent_id="controller",
            run_id=self.run_id,
            action_id=f"act-{uuid.uuid4()}"
        )

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

        # 1. Check skip_if conditions against current state
        skip_conditions: dict[str, bool] = {
            "doc_only": is_doc_only,
            "fast_mode": is_fast_mode,
            "no_test_command": not self.test_command,
        }
        for condition in step.skip_if:
            if skip_conditions.get(condition, False):
                return AgentResult(
                    status="success",
                    agent=step.agent_role,
                    payload={"skipped": True},
                )

        # 2. Dispatch to the appropriate _run_* method based on agent_role
        role = step.agent_role

        if role == "planner":
            return self._run_planner(task, ws_path, failure_context)

        if role == "implementer":
            if is_doc_only:
                # Doc-only path: implementer bypassed, handled in post-processing
                return AgentResult(
                    status="success", agent="implementer",
                    payload={"doc_only": True},
                )
            return self._run_implementer(task, plan or {}, ws_path, tools)

        if role == "testgen":
            self._run_testgen(task, ws_path, is_doc_only)
            return AgentResult(status="success", agent="testgen")

        if role == "debugger":
            test_ok = self._run_fix_loop(task, ws_path, tools, impl or {})
            return AgentResult(
                status="success",
                agent="debugger",
                payload={"test_passing": test_ok},
            )

        if role == "security":
            return self._run_security(task, impl or {}, ws_path)

        if role == "release":
            return self._run_release(task, impl or {}, ws_path, is_fast_mode)

        if role == "archivist":
            return self._run_archivist(
                task, impl or {}, plan or {}, rel or {}, ws_path, is_fast_mode
            )

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

        context = AgentContext(
            task_id=task.task_id,
            run_id=self.run_id,
            objective=objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            constraints=task.constraints,
            acceptance_criteria=task.acceptance_criteria,
            risk_level=task.risk_level,
        )

        raw = self.agents["planner"].run(context)
        self._log_event("plan_created", {
            "steps": len(raw.get("steps", [])),
            "risk": raw.get("risk_level"),
        })

        self._print_plan(raw)

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
        )

        # Keep the user's task constraints, but DROP the JSON formatting constraints
        impl_constraints = list(task.constraints)

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
                
                console.print(f"[bold blue]🔄 Resuming Patch...[/]")
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
        """Helper method: Handle mid-flight delegation requests from the Implementer."""
        # Create a hyper-focused sub-context for the delegated colleague
        sub_context = AgentContext(
            task_id=f"{task.task_id}-delegate-{target}",
            run_id=self.run_id,
            objective=f"Your colleague needs your expertise on a specific sub-task:\n\n{request}",
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            extra={
                "tool_executor": tools,
                "prelude": self._prelude,
                "fast_mode": False,
                "repo_index": getattr(self, "_repo_index", None),
            }
        )
        
        try:
            if target == "security":
                res = self.agents["security"].run(sub_context)
                return f"Verdict: {res.get('verdict')}\nSummary: {res.get('summary')}\nIssues: {res.get('issues', [])}"
            
            elif target == "debugger":
                sub_context.extra["test_command"] = self.test_command
                res = self.agents["debugger"].run(sub_context)
                return f"Diagnosis: {res.get('diagnosis')}\nRoot Cause: {res.get('root_cause')}\nFixes applied: {res.get('fix_summary', 'None')}"
            
            elif target == "testgen":
                sub_context.extra["test_command"] = self.test_command
                res = self.agents["testgen"].run(sub_context)
                return f"Test Generated: {res.get('test_file')}\nDescription: {res.get('description')}"
            
            elif target == "archivist":
                res = self.agents["archivist"].run(sub_context)
                return f"Architecture Notes: {res.get('architecture_notes')}\nADR Written: {res.get('should_write_adr')}"
            
            else:
                return f"Error: Unknown colleague '{target}'."
                
        except Exception as e:
            logger.error(f"Delegation to {target} failed: {e}")
            return f"Colleague {target} encountered an error and could not complete the request: {e}"

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
            self._attest_controller_action(f"TESTGEN {test_file}")
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
                    "error_output": (error_output or "")[:1000],
                    "test_command": self.test_command,
                    "tool_executor": tools, # Hand over the keys to the sandbox
                    "prelude": self._prelude,
                    "repo_index": self._repo_index,
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
        console.print("\n[bold yellow]🕵️  [AUDITOR] Checking for performance smells...[/]")

        diff = self._workspace.diff_full() if self._workspace else ""

        context = AgentContext(
            task_id=task.task_id,
            run_id=self.run_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            previous_output=self._state.to_agent_summary("auditor"),
            extra={
                "diff": diff,
            },
        )

        result = self.auditor.run(context)
        self._log_event("auditor_feedback", {"feedback": result.get("feedback")})
        return result

    def _run_security(self, task: Task, impl: dict, ws_path: Path, is_fast_mode: bool = False) -> AgentResult:
        console.print("\n[bold red]🔒 [FRANKIE] Security scan...[/]")

        diff = self._workspace.diff_full() if self._workspace else ""

        # v2: Structured state, not raw impl blob
        context = AgentContext(
            task_id=task.task_id,
            run_id=self.run_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            previous_output=self._state.to_agent_summary("security"),
            extra={
                "diff": diff,
                "protected_paths": self.config.boundaries.protected_paths,
                "fast_mode": is_fast_mode,
                "repo_index": self._repo_index,  # <--- Add this line to enable query_symbol_map
                "prelude": self._prelude,
            },
        )

        raw = self.agents["security"].run(context)
        result = AgentResult.from_raw(raw)
        self._log_event("security_review", {"verdict": result.payload.get("verdict")})
        return result

    def _run_release(self, task: Task, impl: dict, ws_path: Path, is_fast_mode: bool = False) -> AgentResult:
        console.print("\n[bold cyan]📦 [SEMVER] Release assessment...[/]")

        diff = self._workspace.diff_stat() if self._workspace else ""

        context = AgentContext(
            task_id=task.task_id,
            run_id=self.run_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            previous_output=self._state.to_agent_summary("release"),
            extra={
                "diff": diff,
                "fast_mode": is_fast_mode,
            },
        )

        raw = self.agents["release"].run(context)
        result = AgentResult.from_raw(raw)
        self._log_event("release_assessment", {"bump": result.payload.get("version_bump")})
        return result

    def _run_archivist(
        self, task: Task, impl: dict, plan: dict, release: dict, ws_path: Path, is_fast_mode: bool = False
    ) -> AgentResult:
        """Run Archivist Nova with structured state context."""
        console.print("\n[bold dim]📚 [NOVA] Documenting...[/]")

        existing_docs = []
        for pattern in ["*.md", "docs/**/*.md", "doc/**/*.md"]:
            existing_docs.extend(
                str(p.relative_to(ws_path))
                for p in ws_path.glob(pattern)
                if p.is_file() and ".glitchlab" not in str(p)
            )

        context = AgentContext(
            task_id=task.task_id,
            run_id=self.run_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            previous_output=self._state.to_agent_summary("archivist"),
            extra={
                "existing_docs": existing_docs[:50],
                "fast_mode": is_fast_mode,
            }
        )

        # ── THE MISSING PIECE ──
        # Call Nova's new tool-loop and return the dict result
        raw = self.agents["archivist"].run(context)

        if raw is None:
            return AgentResult(
                status="error",
                payload={
                    "should_write_adr": False,
                    "doc_updates": [],
                    "architecture_notes": "Archivist failed.",
                },
            )

        return AgentResult.from_raw(raw)

    @staticmethod
    def _write_adr(ws_path: Path, adr: dict | str) -> str | None:
        """Write an ADR to the workspace."""
        adr_dir = ws_path / ".context" / "decisions"
        if not adr_dir.exists():
            adr_dir = ws_path / "docs" / "adr"
        adr_dir.mkdir(parents=True, exist_ok=True)

        existing = list(adr_dir.glob("*.md"))
        next_num = len(existing) + 1

        # Handle the case where the LLM returns a raw markdown string
        if isinstance(adr, str):
            title = f"ADR-{next_num:03d}"
            # Try to extract a title from the first markdown header
            first_line = adr.strip().split('\n')[0]
            if first_line.startswith('# '):
                title = first_line.replace('# ', '').strip()
                
            safe_title = re.sub(r'[^a-z0-9\-]+', '-', title.lower())
            safe_title = re.sub(r'-+', '-', safe_title).strip('-')
            filename = f"{next_num:03d}-{safe_title[:50]}.md"
            filepath = adr_dir / filename
            
            filepath.write_text(adr + "\n\n---\n*Generated by GLITCHLAB / Archivist Nova*\n")
            return f"ADR {filepath.relative_to(ws_path)}"

        # Handle the case where the LLM returns a structured dictionary
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
            "task_id": self._state.task_id if self._state else None,
            "data": data or {},
        }
        if self._state:
            self._state.events.append(event)
        logger.debug(f"[EVENT] {event_type}: {data}")