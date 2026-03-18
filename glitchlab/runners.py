"""
GLITCHLAB Agent Runners — Extracted pipeline step executors.

Each function builds an AgentContext, runs the agent, and returns a result.
They are stateless helpers called by the Controller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console

from glitchlab.agents import AgentContext, BaseAgent, AgentResult
from glitchlab.task import Task
from glitchlab.task_state import TaskState

console = Console()


def run_auditor(
    *,
    agent: BaseAgent,
    task: Task,
    ws_path: Path,
    run_id: str,
    repo_path: Path,
    state: TaskState,
    workspace: Any,
) -> dict:
    """Run the auditor agent to check for performance smells."""
    console.print("\n[bold yellow]🕵️  [AUDITOR] Checking for performance smells...[/]")

    diff = workspace.diff_full() if workspace else ""

    context = AgentContext(
        task_id=task.task_id,
        run_id=run_id,
        objective=task.objective,
        repo_path=str(repo_path),
        working_dir=str(ws_path),
        previous_output=state.to_agent_summary("auditor"),
        extra={
            "diff": diff,
        },
    )

    return agent.run(context)


def run_security(
    *,
    agent: BaseAgent,
    task: Task,
    ws_path: Path,
    run_id: str,
    repo_path: Path,
    state: TaskState,
    workspace: Any,
    config: Any,
    repo_index: Any,
    prelude: Any,
    is_fast_mode: bool = False,
) -> AgentResult:
    """Run the security agent to scan for vulnerabilities."""
    console.print("\n[bold red]🔒 [FRANKIE] Security scan...[/]")

    diff = workspace.diff_full() if workspace else ""

    context = AgentContext(
        task_id=task.task_id,
        run_id=run_id,
        objective=task.objective,
        repo_path=str(repo_path),
        working_dir=str(ws_path),
        previous_output=state.to_agent_summary("security"),
        extra={
            "diff": diff,
            "protected_paths": config.boundaries.protected_paths,
            "fast_mode": is_fast_mode,
            "repo_index": repo_index,
            "prelude": prelude,
        },
    )

    raw = agent.run(context, max_steps=5 if is_fast_mode else 15)
    return AgentResult.from_raw(raw)


def run_release(
    *,
    agent: BaseAgent,
    task: Task,
    ws_path: Path,
    run_id: str,
    repo_path: Path,
    state: TaskState,
    workspace: Any,
    is_fast_mode: bool = False,
) -> AgentResult:
    """Run the release agent for version bump assessment."""
    console.print("\n[bold cyan]📦 [SEMVER] Release assessment...[/]")

    diff = workspace.diff_stat() if workspace else ""

    context = AgentContext(
        task_id=task.task_id,
        run_id=run_id,
        objective=task.objective,
        repo_path=str(repo_path),
        working_dir=str(ws_path),
        previous_output=state.to_agent_summary("release"),
        extra={
            "diff": diff,
            "fast_mode": is_fast_mode,
        },
    )

    raw = agent.run(context, max_steps=5 if is_fast_mode else 10)
    return AgentResult.from_raw(raw)


def run_archivist(
    *,
    agent: BaseAgent,
    task: Task,
    ws_path: Path,
    run_id: str,
    repo_path: Path,
    state: TaskState,
    is_fast_mode: bool = False,
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
        run_id=run_id,
        objective=task.objective,
        repo_path=str(repo_path),
        working_dir=str(ws_path),
        previous_output=state.to_agent_summary("archivist"),
        extra={
            "existing_docs": existing_docs[:50],
            "fast_mode": is_fast_mode,
        },
    )

    raw = agent.run(context)

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


def run_delegated_agent(
    *,
    target: str,
    request: str,
    task: Task,
    ws_path: Path,
    run_id: str,
    repo_path: Path,
    agents: dict[str, BaseAgent],
    tools: Any,
    prelude: Any,
    repo_index: Any,
    test_command: str | None = None,
) -> str:
    """Handle mid-flight delegation requests from the Implementer."""
    sub_context = AgentContext(
        task_id=f"{task.task_id}-delegate-{target}",
        run_id=run_id,
        objective=(
            f"Your colleague needs your expertise on a specific sub-task:\n\n{request}"
        ),
        repo_path=str(repo_path),
        working_dir=str(ws_path),
        extra={
            "tool_executor": tools,
            "prelude": prelude,
            "fast_mode": False,
            "repo_index": repo_index,
        },
    )

    try:
        if target == "security":
            res = agents["security"].run(sub_context)
            return (
                f"Verdict: {res.get('verdict')}\n"
                f"Summary: {res.get('summary')}\n"
                f"Issues: {res.get('issues', [])}"
            )

        elif target == "debugger":
            sub_context.extra["test_command"] = test_command
            res = agents["debugger"].run(sub_context)
            return (
                f"Diagnosis: {res.get('diagnosis')}\n"
                f"Root Cause: {res.get('root_cause')}\n"
                f"Fixes applied: {res.get('fix_summary', 'None')}"
            )

        elif target == "testgen":
            sub_context.extra["test_command"] = test_command
            res = agents["testgen"].run(sub_context)
            return (
                f"Test Generated: {res.get('test_file')}\n"
                f"Description: {res.get('description')}"
            )

        elif target == "archivist":
            res = agents["archivist"].run(sub_context)
            return (
                f"Architecture Notes: {res.get('architecture_notes')}\n"
                f"ADR Written: {res.get('should_write_adr')}"
            )

        else:
            return f"Error: Unknown colleague '{target}'."

    except Exception as e:
        logger.error(f"Delegation to {target} failed: {e}")
        return (
            f"Colleague {target} encountered an error and "
            f"could not complete the request: {e}"
        )
