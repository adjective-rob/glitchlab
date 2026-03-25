"""
GLITCHLAB Step Handlers — Per-agent post-processing logic.

Extracted from Controller._execute_pipeline's if/elif chain.
Each handler processes an AgentResult AFTER the agent runs,
mutates PipelineState (the mutable bag of inter-step data),
and returns a HandlerSignal indicating whether to continue, halt, or return early.

To add a new agent type:
  1. Write a handle_<role>_result function
  2. Add it to STEP_HANDLERS
  3. Add the agent_role to your pipeline config

No controller edits required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from loguru import logger
from rich.console import Console
from rich.prompt import Confirm

from glitchlab.agents import AgentResult
from glitchlab.config_loader import PipelineStep
from glitchlab.controller_utils import attest_controller_action
from glitchlab.display import print_security_issues
from glitchlab.doc_inserter import insert_doc_comments, write_adr
from glitchlab.events import emit_event
from glitchlab.governance import BoundaryViolation
from glitchlab.run_context import RunContext
from glitchlab.task import Task, apply_changes, apply_tests
from glitchlab.task_state import StepState

console = Console()


# ---------------------------------------------------------------------------
# Pipeline State — mutable bag passed between handlers within a single run
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """Mutable inter-step data that flows through the pipeline.

    This replaces the 10-element tuple that _execute_pipeline used to return.
    Handlers read and mutate this freely; the controller reads it after the
    pipeline loop completes.
    """

    plan: dict = field(default_factory=dict)
    impl: dict = field(default_factory=dict)
    rel: dict = field(default_factory=dict)
    sec: dict = field(default_factory=dict)
    applied: list[str] = field(default_factory=list)
    is_doc_only: bool = False
    is_fast_mode: bool = False
    test_ok: bool = True
    pipeline_halted: bool = False
    result: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Handler Signal — what the pipeline loop should do after a handler runs
# ---------------------------------------------------------------------------

class HandlerSignal(Enum):
    CONTINUE = auto()     # proceed to next step
    EARLY_RETURN = auto() # stop pipeline, return result as-is


# ---------------------------------------------------------------------------
# Handler type signature
# ---------------------------------------------------------------------------
# All handlers: (ctx, task, step_result, ps) -> HandlerSignal

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def handle_planner_result(
    ctx: RunContext, task: Task, step_result: AgentResult, ps: PipelineState,
) -> HandlerSignal:
    if step_result.status == "error":
        ps.result["status"] = "plan_failed"
        return HandlerSignal.EARLY_RETURN

    ps.plan = step_result.payload

    # Update TaskState with plan output
    ctx.state.plan_steps = [
        StepState(
            step_number=s.get("step_number", 0),
            description=s.get("description", ""),
            files=s.get("files", []),
            action=s.get("action", ""),
            do_not_touch=s.get("do_not_touch", []),
            code_hint=s.get("code_hint", ""),
        )
        for s in ps.plan.get("steps", [])
    ]
    ctx.state.files_in_scope = ps.plan.get("files_likely_affected", [])
    ctx.state.estimated_complexity = ps.plan.get("estimated_complexity", "medium")
    ctx.state.requires_core_change = ps.plan.get("requires_core_change", False)
    ctx.state.mark_phase("plan")
    ctx.state.persist(ctx.ws_path)

    # Boundary Validation (Plan-Level)
    try:
        violations = ctx.boundary.check_plan(ps.plan, ctx.allow_core)
        if violations:
            emit_event(ctx, "core_override", {"files": violations})
            console.print(f"[yellow]⚠ Core override granted for: {violations}[/]")
    except BoundaryViolation as e:
        console.print(f"[red]🚫 {e}[/]")
        emit_event(ctx, "boundary_violation", {"error": str(e)})
        ps.result["status"] = "boundary_violation"
        return HandlerSignal.EARLY_RETURN

    # Governance Mode Routing
    is_evolution = task.mode == "evolution"
    if is_evolution:
        ctx.config.intervention.pause_before_pr = True

    # Strict doc-only detection
    is_maintenance = task.mode == "maintenance"
    objective_lower = task.objective.lower()
    ps.is_doc_only = (
        is_maintenance
        and any(term in objective_lower for term in ["doc", "documentation", "///"])
        and task.risk_level == "low"
        and all(s.get("action") == "modify" for s in ps.plan.get("steps", []))
        and any(f.endswith(".rs") for f in ps.plan.get("files_likely_affected", []))
    )

    return HandlerSignal.CONTINUE


# ---------------------------------------------------------------------------
# Implementer
# ---------------------------------------------------------------------------

def handle_implementer_result(
    ctx: RunContext, task: Task, step_result: AgentResult, ps: PipelineState,
) -> HandlerSignal:
    is_maintenance = task.mode == "maintenance"

    # Maintenance: Surgical Documentation Path
    if ps.is_doc_only:
        console.print(
            "\n[bold dim]📄 [MAINTENANCE MODE] "
            "Surgical documentation update — implementer bypassed.[/]"
        )
        ps.impl = {
            "changes": [],
            "tests_added": [],
            "commit_message": f"docs: update documentation for {task.task_id}",
            "summary": "Surgical documentation insertion.",
        }
        ps.applied = []

        for f in ps.plan.get("files_likely_affected", []):
            fpath = ctx.ws_path / f
            if fpath.exists():
                inserted = insert_doc_comments(fpath, ctx.router)
                if inserted:
                    ps.applied.append(f"DOC {f}")

        for entry in ps.applied:
            console.print(f"  [cyan]{entry}[/]")
            attest_controller_action(entry, ctx.run_id)

    # Standard Execution Path
    else:
        if step_result.status == "error":
            ps.result["status"] = "implementation_failed"
            return HandlerSignal.EARLY_RETURN
        ps.impl = step_result.payload

        # Update TaskState with implementation output
        ctx.state.files_modified = [
            c.get("file", "")
            for c in ps.impl.get("changes", [])
            if c.get("action") in ("modify", "create")
        ]
        ctx.state.files_created = [
            c.get("file", "")
            for c in ps.impl.get("changes", [])
            if c.get("action") == "create"
        ]
        ctx.state.tests_added = [
            t.get("file", "") for t in ps.impl.get("tests_added", [])
        ]
        ctx.state.commit_message = ps.impl.get("commit_message", "")
        ctx.state.implementation_summary = ps.impl.get("summary", "")
        ctx.state.mark_phase("implement")
        ctx.state.persist(ctx.ws_path)

        is_high_complexity = ps.plan.get(
            "estimated_complexity", ""
        ).lower() in ["high", "large", "unknown"]
        if is_high_complexity:
            console.print("  [dim]High complexity: allowing full-file rewrites.[/]")

        try:
            ps.applied = apply_changes(
                ctx.ws_path,
                ps.impl.get("changes", []),
                boundary=ctx.boundary,
                allow_core=ctx.allow_core,
                allow_test_modifications=not is_maintenance,
                allow_full_rewrite=True,
            )
            ps.applied += apply_tests(
                ctx.ws_path,
                ps.impl.get("tests_added", []),
                allow_test_modifications=not is_maintenance,
            )
        except BoundaryViolation as e:
            console.print(f"[red]🚫 Boundary Violation during implementation: {e}[/]")
            ps.result["status"] = "boundary_violation"
            return HandlerSignal.EARLY_RETURN

        for entry in ps.applied:
            console.print(f"  [cyan]{entry}[/]")
            attest_controller_action(entry, ctx.run_id)

    # Patch & Surgical failure retry (one attempt)
    # Catch BOTH "FAIL" (surgical) and "PATCH_FAILED" (diffs)
    patch_failures = [a for a in ps.applied if "FAIL" in a or "PATCH_FAILED" in a]

    if patch_failures:
        console.print(
            "[yellow]⚠ Edit failed to apply "
            "(likely a whitespace mismatch). "
            "Attempting one auto-repair...[/]"
        )

        for entry in ps.applied:
            console.print(f"  [cyan]{entry}[/]")
            attest_controller_action(entry, ctx.run_id)

        if any("FAIL" in a or "PATCH_FAILED" in a for a in ps.applied):
            console.print(
                "[red]❌ Auto-repair failed. "
                "Aborting to prevent corrupted PR.[/]"
            )
            ps.result["status"] = "implementation_failed"
            return HandlerSignal.EARLY_RETURN

        # Import here to avoid circular — retry_patch lives in agent_runners
        from glitchlab.agent_runners import retry_patch

        retry_result = retry_patch(ctx, task, ps.plan, ps.impl, ps.applied)

        if retry_result.status == "error":
            ps.result["status"] = "implementation_failed"
            return HandlerSignal.EARLY_RETURN

        ps.applied = apply_changes(
            ctx.ws_path,
            retry_result.payload.get("changes", []),
            boundary=ctx.boundary,
            allow_core=ctx.allow_core,
            allow_test_modifications=not is_maintenance,
            allow_full_rewrite=True,
        )

        for entry in ps.applied:
            console.print(f"  [cyan]{entry}[/]")
            attest_controller_action(entry, ctx.run_id)

        if any(a.startswith("PATCH_FAILED") for a in ps.applied):
            console.print("[red]❌ Patch retry failed. Aborting.[/]")
            ps.result["status"] = "implementation_failed"
            return HandlerSignal.EARLY_RETURN

    # Compute fast_mode for downstream agents
    ps.is_fast_mode = (
        len(ctx.state.files_modified) <= 2
        and ctx.state.estimated_complexity in ("trivial", "small")
    )
    if ps.is_fast_mode:
        console.print(
            "  [dim]Trivial change detected. "
            "Forcing downstream agents into Fast Mode.[/]"
        )

    return HandlerSignal.CONTINUE


# ---------------------------------------------------------------------------
# Debugger
# ---------------------------------------------------------------------------

def handle_debugger_result(
    ctx: RunContext, task: Task, step_result: AgentResult, ps: PipelineState,
) -> HandlerSignal:
    ps.test_ok = step_result.payload.get("test_passing", True)

    if not ps.test_ok:
        ps.result["status"] = "tests_failed"
        console.print("[red]❌ Fix loop exhausted. Tests still failing.[/]")
        if not _confirm(ctx, "Continue to PR anyway?"):
            return HandlerSignal.EARLY_RETURN

    ctx.state.test_passing = ps.test_ok
    ctx.state.mark_phase("test")
    ctx.state.persist(ctx.ws_path)

    return HandlerSignal.CONTINUE


# ---------------------------------------------------------------------------
# Testgen
# ---------------------------------------------------------------------------

def handle_testgen_result(
    ctx: RunContext, task: Task, step_result: AgentResult, ps: PipelineState,
) -> HandlerSignal:
    # Testgen has no post-processing — all work done in the runner
    return HandlerSignal.CONTINUE


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

def handle_security_result(
    ctx: RunContext, task: Task, step_result: AgentResult, ps: PipelineState,
) -> HandlerSignal:
    ps.sec = step_result.payload
    ctx.state.mark_phase("security")

    emit_event(ctx, "security_review", {
        "verdict": ps.sec.get("verdict"),
        "issues_count": len(ps.sec.get("issues", [])),
    }, agent_id="security")

    if ps.sec.get("verdict") == "block":
        ctx.state.security_verdict = "block"
        console.print("[red]🚫 Security blocked this change.[/]")
        print_security_issues(ps.sec)

        if ctx.auto_approve:
            console.print("[red]❌ Auto-approve enabled. Aborting dangerous PR.[/]")
            ps.result["status"] = "security_blocked"
            return HandlerSignal.EARLY_RETURN

        if not _confirm(ctx, "Override security block?"):
            ps.result["status"] = "security_blocked"
            return HandlerSignal.EARLY_RETURN
    else:
        # Normalize missing/empty verdict to "warn"
        verdict = ps.sec.get("verdict") or "warn"
        ctx.state.security_verdict = verdict

        if verdict == "warn":
            console.print("[yellow]⚠ Security review returned warnings.[/]")
            print_security_issues(ps.sec)

        # Prevent the generic error-halt from aborting
        # the pipeline on a non-blocking verdict.
        step_result.status = "success"

    return HandlerSignal.CONTINUE


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------

def handle_release_result(
    ctx: RunContext, task: Task, step_result: AgentResult, ps: PipelineState,
) -> HandlerSignal:
    ps.rel = step_result.payload
    ctx.state.version_bump = ps.rel.get("version_bump", "")
    ctx.state.changelog_entry = ps.rel.get("changelog_entry", "")
    ctx.state.mark_phase("release")

    emit_event(ctx, "release_assessment", {
        "version_bump": ps.rel.get("version_bump"),
        "changelog_entry": ps.rel.get("changelog_entry", "")[:200],
    }, agent_id="release")

    return HandlerSignal.CONTINUE


# ---------------------------------------------------------------------------
# Archivist
# ---------------------------------------------------------------------------

def handle_archivist_result(
    ctx: RunContext, task: Task, step_result: AgentResult, ps: PipelineState,
) -> HandlerSignal:
    nova_result = step_result.payload
    is_maintenance = task.mode == "maintenance"

    if is_maintenance:
        nova_result["should_write_adr"] = False

    if (
        nova_result
        and nova_result.get("should_write_adr")
        and nova_result.get("adr")
    ):
        adr_applied = write_adr(ctx.ws_path, nova_result["adr"])
        if adr_applied:
            console.print(f"  [cyan]{adr_applied}[/]")
            attest_controller_action(adr_applied, ctx.run_id)

    # Maintenance mode: forbid file create/delete and out-of-scope edits
    if is_maintenance:
        allowed_paths = set(ps.plan.get("files_likely_affected") or [])
        if not allowed_paths:
            raise RuntimeError("Maintenance mode requires explicit files_likely_affected")

        for mpath in allowed_paths:
            ctx.workspace._git("add", mpath, check=False)

        diff_output = ctx.workspace._git("diff", "--cached", "--name-status", check=False)
        lines = diff_output.splitlines() if diff_output else []

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

        out_of_scope = [p for p in touched if p not in allowed_paths]
        if created or deleted or out_of_scope:
            raise RuntimeError(
                f"Maintenance violation. "
                f"created={created} deleted={deleted} "
                f"out_of_scope={out_of_scope}"
            )

    return HandlerSignal.CONTINUE


# ---------------------------------------------------------------------------
# Handler Registry
# ---------------------------------------------------------------------------

STEP_HANDLERS: dict[str, Any] = {
    "planner": handle_planner_result,
    "implementer": handle_implementer_result,
    "debugger": handle_debugger_result,
    "testgen": handle_testgen_result,
    "security": handle_security_result,
    "release": handle_release_result,
    "archivist": handle_archivist_result,
}


def register_handler(role: str):
    """Decorator to register a custom step handler for a new agent role."""
    def decorator(fn):
        STEP_HANDLERS[role] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confirm(ctx: RunContext, prompt: str) -> bool:
    if ctx.auto_approve:
        return True
    return Confirm.ask(f"[bold]{prompt}[/]")