"""
GLITCHLAB Lifecycle — Pre/post pipeline operations.

Extracted from Controller: startup, finalize, repo checks, PR creation,
session entry writing, and the rebase-before-PR flow.

These are pure git/workspace/PR mechanics with no agent logic.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from glitchlab.controller_utils import calculate_quality_score
from glitchlab.display import build_pr_body, print_budget_summary
from glitchlab.event_bus import bus
from glitchlab.events import emit_event
from glitchlab.indexer import build_index
from glitchlab.run_context import RunContext
from glitchlab.scope import ScopeResolver
from glitchlab.task import Task
from glitchlab.task_state import DirtyRepoError
from glitchlab.workspace import Workspace
from glitchlab.workspace.tools import ToolExecutor

console = Console()


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def print_banner(task: Task) -> None:
    """Print the task banner."""
    console.print(Panel(
        f"[bold green]Task:[/] {task.objective[:120]}\n"
        f"[bold]ID:[/] {task.task_id}  |  [bold]Source:[/] {task.source}\n"
        f"[bold]Risk:[/] {task.risk_level}  |  [bold]Mode:[/] {task.mode.upper()}",
        title="⚡ GLITCHLAB v4.3.1",
        subtitle="Build Weird. Ship Clean.",
        border_style="bright_green",
    ))


# ---------------------------------------------------------------------------
# Repo Check
# ---------------------------------------------------------------------------

def check_repo_clean(repo_path: Path) -> None:
    """Raise DirtyRepoError if the repo has uncommitted changes or is behind remote."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
        capture_output=True,
        text=True,
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
        subprocess.run(["git", "fetch", "--quiet"], cwd=repo_path, timeout=10)
        behind = subprocess.run(
            ["git", "rev-list", "HEAD..@{u}", "--count"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if behind and behind.isdigit() and int(behind) > 0:
            console.print(
                f"[red]🚫 Cannot run: Local branch is behind remote by {behind} commits. "
                f"Please pull changes.[/]"
            )
            raise DirtyRepoError("Local branch is behind remote.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def startup(
    ctx: RunContext,
    task: Task,
) -> str:
    """Create workspace, build indexes, load constraints.

    Mutates ctx in-place (sets ws_path, workspace, tools, scope, repo_index).
    Returns failure_context string for the planner.
    """
    # 1. Create workspace
    ctx.workspace = Workspace(
        ctx.repo_path, task.task_id,
        ctx.config.workspace.worktree_dir,
    )
    ws_path = ctx.workspace.create()
    ctx.ws_path = ws_path
    emit_event(ctx, "workspace_created", {"path": str(ws_path)})

    ctx.tools = ToolExecutor(
        allowed_tools=ctx.config.allowed_tools,
        blocked_patterns=ctx.config.blocked_patterns,
        working_dir=ws_path,
    )

    # 1.5. Build repo index (file map for planner)
    console.print("\n[bold dim]🗂  [INDEX] Scanning repository...[/]")
    ctx.repo_index = build_index(ws_path)
    console.print(
        f"  [dim]{ctx.repo_index.total_files} files, "
        f"{len(ctx.repo_index.languages)} languages[/]"
    )
    emit_event(ctx, "repo_indexed", {
        "total_files": ctx.repo_index.total_files,
        "languages": ctx.repo_index.languages,
    })

    # 1.6. Initialize ScopeResolver (Layer 1)
    ctx.scope = ScopeResolver(ws_path, ctx.repo_index)

    # 1.7. Prelude: load constraints only (not global prefix)
    if ctx.prelude.available:
        console.print("[bold dim]📋 [PRELUDE] Loading constraints...[/]")
        ctx.prelude.refresh()
        prelude_constraints = ctx.prelude.get_constraints()
        if prelude_constraints:
            task.constraints = list(set(task.constraints + prelude_constraints))
            console.print(f"  [dim]{len(prelude_constraints)} constraints merged[/]")
        emit_event(ctx, "prelude_constraints_loaded", {
            "count": len(prelude_constraints) if prelude_constraints else 0,
        })

    # 1.8. Load failure context from history
    failure_context = ctx.history.build_failure_context()
    if failure_context:
        console.print("  [dim]Loaded recent failure patterns for planner[/]")

    return failure_context


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

def finalize(
    ctx: RunContext,
    task: Task,
    plan: dict,
    impl: dict,
    rel: dict,
    sec: dict,
    is_doc_only: bool,
    is_fast_mode: bool,
    result: dict,
    pipeline_halted: bool,
) -> dict:
    """Commit changes, create PR, archive task. Returns updated result dict."""
    # Phase routing: doc-only defaults for downstream
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

    # 8. Commit + PR
    ctx.state.mark_phase("commit")
    ctx.state.persist(ctx.ws_path)

    commit_msg = impl.get("commit_message", f"glitchlab: {task.task_id}")
    ctx.workspace.commit(commit_msg)

    # Rebase Before PR
    if getattr(ctx.config, "automation", None) and getattr(ctx.config.automation, "rebase_before_pr", False):
        console.print("[dim]🔄 Rebasing onto origin/main to prevent conflicts...[/]")

        if not ctx.workspace.rebase(auto_abort=False):
            resolved = False
            if ctx.test_command:
                console.print(
                    "[yellow]⚠️ Rebase conflict detected. "
                    "Handing over to Debugger for auto-resolution...[/]"
                )
                from glitchlab.agent_runners import run_fix_loop
                resolved = run_fix_loop(ctx, task, impl)

            if not resolved:
                ctx.workspace._worktree_git("rebase", "--abort", check=False)
                result["status"] = "rebase_conflict"
                console.print(
                    "[red]❌ Auto-resolution failed or no tests available. PR aborted.[/]"
                )
                return result
            else:
                ctx.workspace._worktree_git("add", "-A")
                env = os.environ.copy()
                env["GIT_EDITOR"] = "true"
                subprocess.run(
                    ["git", "rebase", "--continue"],
                    cwd=ctx.ws_path, env=env, check=False,
                )
                console.print("[bold green]✅ Rebase conflict auto-resolved by agent![/]")

    if getattr(ctx.config.intervention, "pause_before_pr", True):
        diff = ctx.workspace.diff_stat()
        console.print(Panel(diff, title="Diff Summary", border_style="cyan"))
        if not _confirm(ctx, "Create PR?"):
            result["status"] = "pr_cancelled"
            return result

    try:
        ctx.workspace.push(force=True)
        pr_url = _create_pr(ctx, task, impl, rel)
        result["pr_url"] = pr_url
        result["status"] = "pr_created"
        console.print(f"[bold green]✅ PR created: {pr_url}[/]")

        # Auto-Merge
        if getattr(ctx.config, "automation", None) and getattr(ctx.config.automation, "auto_merge_pr", False):
            console.print("[dim]🚀 Auto-merge enabled. Squashing and merging...[/]")
            merge_res = subprocess.run(
                ["gh", "pr", "merge", pr_url, "--squash"],
                cwd=ctx.repo_path,
                capture_output=True,
                text=True,
            )

            if merge_res.returncode == 0:
                result["status"] = "merged"
                console.print("[bold green]🎉 PR Auto-Merged successfully![/]")
            else:
                console.print(
                    f"[yellow]⚠️ Auto-merge blocked by GitHub (likely awaiting CI). "
                    f"PR remains open.\n{merge_res.stderr.strip()}[/]"
                )

    except Exception as e:
        logger.warning(f"PR creation failed: {e}")
        result["status"] = "committed"
        result["branch"] = ctx.workspace.branch_name
        console.print(f"[yellow]Changes committed to {ctx.workspace.branch_name}[/]")

    print_budget_summary(ctx.router.budget.summary())
    result["events"] = ctx.state.events
    result["budget"] = ctx.router.budget.summary()

    if getattr(task, "file_path", None) and task.file_path.exists() and task.file_path.parent.name == "queue":
        archive_dir = task.file_path.parent.with_name("archive")
        archive_dir.mkdir(parents=True, exist_ok=True)
        task.file_path.rename(archive_dir / task.file_path.name)
        console.print(f"[dim]Moved task file to {archive_dir / task.file_path.name}[/]")

    return result


# ---------------------------------------------------------------------------
# Post-run (quality scoring, event emission, session entry)
# ---------------------------------------------------------------------------

def post_run(
    ctx: RunContext,
    task: Task,
    result: dict,
) -> dict:
    """Quality scoring, Zephyr event emission, session entry, cleanup."""
    quality_score = calculate_quality_score(
        ctx.router.budget.summary(), ctx.state
    )
    result["quality_score"] = quality_score

    bus.emit(
        event_type="run.completed",
        payload=result,
        run_id=ctx.run_id,
        metadata={"quality_score": quality_score},
    )

    return result


def write_session_entry(ctx: RunContext, task: Task, result: dict[str, Any]) -> None:
    """Write a Prelude-compatible session entry to .context/session.json."""
    session_file = ctx.repo_path / ".context" / "session.json"
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

    entry = {
        "id": ctx.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "prompt",
        "summary": task.objective[:200],
        "filesAffected": list(set(
            (ctx.state.files_modified if ctx.state else [])
            + (ctx.state.files_created if ctx.state else [])
        )),
        "outcome": outcome,
        "tags": [ctx.state.mode if ctx.state else "unknown"],
    }

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
                    "entries": [],
                }],
            }

        entries = data.get("sessions", [{}])[0].get("entries", [])
        entries.append(entry)

        # Cap at 20 entries
        if len(entries) > 20:
            entries = entries[-20:]

        data["sessions"][0]["entries"] = entries
        session_file.write_text(json.dumps(data, indent=2))
        logger.debug(f"[PRELUDE] Session entry written: {outcome} — {task.objective[:60]}")
    except Exception as e:
        logger.warning(f"[PRELUDE] Failed to write session entry: {e}")


# ---------------------------------------------------------------------------
# PR Creation
# ---------------------------------------------------------------------------

def _create_pr(ctx: RunContext, task: Task, impl: dict, release: dict) -> str:
    """Create a GitHub PR via gh CLI."""
    title = impl.get("commit_message", f"glitchlab: {task.task_id}")
    body = build_pr_body(task, impl, release)

    result = subprocess.run(
        [
            "gh", "pr", "create",
            "--title", title,
            "--body", body,
            "--head", ctx.workspace.branch_name,
        ],
        cwd=ctx.repo_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"PR creation failed: {result.stderr}")

    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confirm(ctx: RunContext, prompt: str) -> bool:
    if ctx.auto_approve:
        return True
    return Confirm.ask(f"[bold]{prompt}[/]")