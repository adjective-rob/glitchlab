"""
GLITCHLAB Workspace Isolation (v2.1)

Transactional sandboxing. Uses 'git worktree' to ensure that
each agent has a completely isolated view of the filesystem,
preventing parallel state bleed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from loguru import logger


class WorkspaceError(Exception):
    pass


class Workspace:
    """
    Manages an isolated git worktree for a single task.
    """

    def __init__(self, repo_path: Path, task_id: str, worktree_base: str = ".glitchlab/worktrees"):
        self.repo_path = repo_path.resolve()
        self.task_id = task_id
        self.branch_name = f"glitchlab/{task_id}"
        self.worktree_path = self.repo_path / worktree_base / task_id
        self._created = False

    @property
    def path(self) -> Path:
        return self.worktree_path

    def create(self, base_branch: str = "main") -> Path:
        """
        Creates a transactional sandbox. 
        If a stale worktree or branch exists, it resets them.
        """
        # 1. Cleanup stale state if it exists from a crashed previous run
        if self.worktree_path.exists() or self._branch_exists(self.branch_name):
            logger.warning(f"[WORKSPACE] Found stale state for {self.task_id}. Resetting...")
            self.cleanup()

        # 2. Ensure parent structure
        self.worktree_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Transactional Creation: Create branch and worktree in one shot
        # We use -B to force create/reset the branch to the base_branch (main)
        try:
            # -B creates or resets the branch to the start-point
            self._git("worktree", "add", "-B", self.branch_name, str(self.worktree_path), base_branch)
            self._created = True
            logger.info(f"[WORKSPACE] Transactional sandbox created: {self.worktree_path}")
        except Exception as e:
            raise WorkspaceError(f"Failed to create isolated worktree: {e}")

        return self.worktree_path

    def diff_full(self) -> str:
        """Get the full unified diff of all changes in the worktree."""
        self._worktree_git("add", "-A", check=False)
        return self._worktree_git("diff", "--cached", capture=True, check=False)

    def diff_stat(self) -> str:
        """Get the diff stat of changes (staged or latest commit)."""
        self._worktree_git("add", "-A", check=False)
        diff = self._worktree_git("diff", "--cached", "--stat", capture=True, check=False)
        
        if not diff.strip():
            # If already committed, show the stats for the commit we just made
            diff = self._worktree_git("show", "--stat", "--format=", "HEAD", capture=True, check=False)
            
        return diff.strip()

    def commit(self, message: str, add_all: bool = True) -> str | None:
        """Stage and commit changes inside the worktree."""
        if add_all:
            self._worktree_git("add", "-A")

        result = self._worktree_git("status", "--porcelain", capture=True)
        if not result.strip():
            logger.info("[WORKSPACE] Nothing to commit.")
            return None

        self._worktree_git("commit", "-m", message)
        sha = self._worktree_git("rev-parse", "HEAD", capture=True).strip()
        return sha

    def push(self, force: bool = True) -> None:
        """Push the task branch. Force push by default for transactional integrity."""
        cmd = ["push", "-u", "origin", self.branch_name]
        if force:
            cmd.insert(1, "--force")
        
        self._worktree_git(*cmd)
        logger.info(f"[WORKSPACE] Pushed (force={force}): {self.branch_name}")

    def cleanup(self) -> None:
        """Hard removal of worktree and pruning of git metadata."""
        # 1. Remove the worktree from git's tracking
        try:
            self._git("worktree", "remove", "--force", str(self.worktree_path), check=False)
        except Exception:
            pass

        # 2. Force delete the directory (handles cases where git remove fails)
        if self.worktree_path.exists():
            shutil.rmtree(self.worktree_path, ignore_errors=True)

        # 3. Delete the local branch to keep the repo index clean
        try:
            self._git("branch", "-D", self.branch_name, check=False)
        except Exception:
            pass

        # 4. Final pruning
        self._git("worktree", "prune", check=False)
        self._created = False
        logger.info(f"[WORKSPACE] Transactional cleanup complete: {self.task_id}")

    def _branch_exists(self, name: str) -> bool:
        """Check if a local branch exists."""
        res = self._git("branch", "--list", name, capture=True)
        return name in res

    def _git(self, *args: str, check: bool = True, capture: bool = False) -> str:
        return self._run_cmd(["git", *args], cwd=self.repo_path, check=check, capture=capture)

    def _worktree_git(self, *args: str, check: bool = True, capture: bool = False) -> str:
        return self._run_cmd(["git", *args], cwd=self.worktree_path, check=check, capture=capture)

    @staticmethod
    def _run_cmd(cmd: list[str], cwd: Path, check: bool = True, capture: bool = False) -> str:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
        if check and result.returncode != 0:
            raise WorkspaceError(f"Git failed: {' '.join(cmd)}\n{result.stderr}")
        return result.stdout if capture else ""