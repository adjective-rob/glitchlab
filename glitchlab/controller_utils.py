"""
GLITCHLAB Controller Utilities — Git helpers, quality scoring, attestation.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from glitchlab.event_bus import bus
from glitchlab.task_state import TaskState


def is_git_repo(path: Path) -> bool:
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


def run_git(
    args: list[str], cwd: Path, timeout: int = 20
) -> subprocess.CompletedProcess:
    """Run a git command and capture output for logging."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def pre_task_git_fetch(repo_path: Path) -> None:
    """Best-effort fetch to ensure planning is against recent `origin/main`.

    Soft-fails (warn + continue) to avoid breaking offline/CI runs.
    """
    if not is_git_repo(repo_path):
        logger.debug(f"[GIT] Skipping fetch: not a git repo: {repo_path}")
        return

    try:
        res = run_git(["fetch", "origin", "main"], cwd=repo_path, timeout=20)
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


def calculate_quality_score(
    budget_summary: dict, state: TaskState | None
) -> dict[str, Any]:
    """Calculate a run quality score out of 100 based on efficiency and convergence."""
    score = 100

    # 1. Time & Efficiency (Penalize excessive token usage)
    total_tokens = budget_summary.get("total_tokens", 0)
    if total_tokens > 50000:
        score -= min(30, (total_tokens - 50000) // 5000)  # Max penalty 30

    # 2. Convergence (Did it struggle in the fix loop?)
    debug_attempts = 0
    if state:
        debug_attempts = state.debug_attempts
        if debug_attempts > 0:
            score -= debug_attempts * 10  # Heavy penalty for needing multiple fix attempts

    return {
        "score": max(0, score),
        "tokens_used": total_tokens,
        "debug_attempts": debug_attempts,
    }


def attest_controller_action(
    action_summary: str, run_id: str
) -> None:
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
            "allowed": True,
        },
        agent_id="controller",
        run_id=run_id,
        action_id=f"act-{uuid.uuid4()}",
    )
