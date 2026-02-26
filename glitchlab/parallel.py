"""
GLITCHLAB Parallel Runner (v2.1)

Transactional parallel execution. Each worker:
  1. Creates a dedicated Git worktree from a fresh 'main'.
  2. Runs the controller loop in total isolation.
  3. Commits and pushes to its own unique task branch.

This prevents race conditions and stale index planning.
"""

from __future__ import annotations

import concurrent.futures
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


def _run_single_task(
    task_file: Path, 
    repo_path: Path, 
    allow_core: bool, 
    test_command: str | None
) -> dict[str, Any]:
    """
    Transactional Task Worker. 
    Enforces isolation through sandboxed git worktrees.
    """
    try:
        from glitchlab.config_loader import load_config
        from glitchlab.controller import Controller, Task
        
        # v2.1: Each worker loads its own fresh config and task
        config = load_config(repo_path)
        task = Task.from_yaml(task_file)
        
        # The Controller's internal Workspace will now handle 
        # the 'git worktree add' to ensure branch isolation.
        controller = Controller(
            repo_path=repo_path,
            config=config,
            allow_core=allow_core,
            auto_approve=True,  # Mandatory for batch/parallel
            test_command=test_command,
        )
        
        # Controller.run() manages the isolated worktree lifecycle
        return controller.run(task)
        
    except Exception as e:
        logger.error(f"[PARALLEL] Transaction failed: {task_file.name} — {e}")
        return {
            "task_id": task_file.stem,
            "status": "error",
            "error": str(e),
        }


def run_parallel(
    repo_path: Path,
    task_files: list[Path],
    max_workers: int = 3,
    allow_core: bool = False,
    auto_approve: bool = True,
    test_command: str | None = None,
) -> list[dict[str, Any]]:
    """
    Orchestrate sandboxed task execution.
    """
    repo_path = repo_path.resolve()

    _print_parallel_header(len(task_files), max_workers)

    # Policy Override: Parallel mode MUST be transactional
    if not auto_approve:
        auto_approve = True

    results: list[dict[str, Any]] = []

    # Use ProcessPool to ensure that file system locks and 
    # Git environment variables don't bleed between tasks.
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_run_single_task, tf, repo_path, allow_core, test_command): tf
            for tf in task_files
        }

        for future in concurrent.futures.as_completed(future_to_file):
            task_file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                _log_task_completion(result, task_file)
            except Exception as e:
                results.append({
                    "task_id": task_file.stem,
                    "status": "error",
                    "error": str(e),
                })

    _print_parallel_summary(results)
    return results

# --- Helpers ---

def _print_parallel_header(count: int, workers: int):
    console.print(f"\n[bold]⚡ GLITCHLAB Transactional Mode — {count} tasks, {workers} workers[/]")
    console.print("[dim]Each task is isolated in a unique Git worktree sandbox.[/]\n")

def _log_task_completion(result: dict, task_file: Path):
    status = result.get("status", "unknown")
    task_id = result.get("task_id", task_file.stem)
    color = "green" if status in ("pr_created", "committed") else "red"
    console.print(f"  [{color}]{task_id}: {status}[/]")

def _print_parallel_summary(results: list[dict]) -> None:
    """Consolidated report of the transactional batch."""
    table = Table(title="Parallel Run Results", border_style="bright_green")
    table.add_column("Task")
    table.add_column("Status")
    table.add_column("PR / Branch")
    table.add_column("Cost")

    for r in results:
        status = r.get("status", "unknown")
        color = {"pr_created": "green", "committed": "yellow"}.get(status, "red")
        pr = r.get("pr_url", r.get("branch", "—"))
        budget = r.get("budget", {})
        cost = f"${budget.get('estimated_cost', 0):.4f}"

        table.add_row(r.get("task_id", "?"), f"[{color}]{status}[/]", str(pr)[:60], cost)

    console.print(table)
    
    total_cost = sum(r.get("budget", {}).get("estimated_cost", 0) for r in results)
    successes = sum(1 for r in results if r.get("status") in ("pr_created", "committed"))
    console.print(f"\n[bold]{successes}/{len(results)} completed | Total cost: ${total_cost:.4f}[/]")