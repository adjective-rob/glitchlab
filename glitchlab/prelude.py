"""
GLITCHLAB ← Prelude Integration (v1.2.0+)

Bridges prelude-context (machine-readable codebase context)
into the GLITCHLAB agent pipeline so every agent understands
the project's stack, architecture, patterns, constraints, and
decisions before it plans or writes a single line.

Prelude is the memory. GLITCHLAB is the muscle.

v1.2.0 changes:
  - Prelude can now be called from outside the target repo
  - Repo path passed as positional argument: `prelude export ~/path/to/repo`
  - External brain path set via PRELUDE_ROOT env var
  - No --brain flag — environment variable only

Usage:
    prelude = PreludeContext(repo_path)
    if prelude.available:
        prelude.refresh()
        prefix = prelude.build_agent_prefix()
        summary = prelude.summary()
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

# Minimum supported Prelude version
MIN_PRELUDE_VERSION = "1.2.0"


class PreludeVersionError(RuntimeError):
    """Raised when the installed Prelude version is too old."""
    pass


class PreludeContext:
    """
    Interface to Prelude's .context/ directory and CLI (v1.2.0+).

    Three modes of operation:
      1. Full CLI (v1.2.0+) available → can init, export, update
      2. .context/ exists but no CLI → read-only (uses cached files)
      3. Neither → gracefully disabled, agents get no project context

    v1.2.0 key behavior:
      - All CLI calls pass repo_path as a positional argument
      - Never changes cwd — works from any directory
      - PRELUDE_ROOT env var controls external brain location
    """

    def __init__(self, repo_path: Path, brain_path: Path | None = None):
        """
        Args:
            repo_path:  The repository to generate context for.
            brain_path: Optional external brain directory (sets PRELUDE_ROOT).
                        Defaults to ~/.glitchlab/brain if not specified.
        """
        self.repo_path = repo_path.resolve()
        self.brain_path = (brain_path or Path.home() / ".glitchlab" / "brain").resolve()
        self.context_dir = self.repo_path / ".context"
        self._cli_path = shutil.which("prelude")
        self._cached_export: str | None = None
        self._version_checked = False

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def cli_available(self) -> bool:
        """Is the prelude CLI installed?"""
        return self._cli_path is not None

    @property
    def context_exists(self) -> bool:
        """Does a .context/ directory exist in this repo?"""
        return self.context_dir.is_dir()

    @property
    def available(self) -> bool:
        """Can we get any context at all?"""
        return self.cli_available or self.context_exists

    # ------------------------------------------------------------------
    # Version Enforcement
    # ------------------------------------------------------------------

    def get_version(self) -> str | None:
        """Return the installed Prelude version string, or None if not installed."""
        if not self.cli_available:
            return None
        try:
            result = subprocess.run(
                ["prelude", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except Exception:
            return None

    def assert_version(self, min_version: str = MIN_PRELUDE_VERSION) -> None:
        """
        Raise PreludeVersionError if installed version is below min_version.
        Call once at startup to prevent subtle breakage from old CLI versions.
        """
        version = self.get_version()
        if version is None:
            raise PreludeVersionError(
                f"Prelude CLI not found. Install with: npm install -g prelude-context"
            )
        if self._version_tuple(version) < self._version_tuple(min_version):
            raise PreludeVersionError(
                f"Prelude {min_version}+ required. Found {version}. "
                f"Upgrade with: npm install -g prelude-context"
            )
        logger.debug(f"[PRELUDE] Version {version} OK (>= {min_version})")

    def check_version(self, min_version: str = MIN_PRELUDE_VERSION) -> bool:
        """
        Like assert_version but returns False instead of raising.
        Use when version mismatch should be a warning, not a hard stop.
        """
        try:
            self.assert_version(min_version)
            return True
        except PreludeVersionError as e:
            logger.warning(f"[PRELUDE] {e}")
            return False

    # ------------------------------------------------------------------
    # CLI Operations
    # ------------------------------------------------------------------

    def init(self, force: bool = False) -> bool:
        """
        Run `prelude init <repo_path>` to analyze the repo and create .context/.

        Args:
            force: Pass --force to overwrite existing .context/ directory.
        """
        if not self.cli_available:
            logger.warning("[PRELUDE] CLI not installed. Run: npm install -g prelude-context")
            return False

        args = ["init", str(self.repo_path)]
        if force:
            args.append("--force")

        result = self._run(*args)
        if result.returncode == 0:
            logger.info("[PRELUDE] Initialized .context/ directory")
            return True
        else:
            logger.error(f"[PRELUDE] Init failed: {result.stderr}")
            return False

    def update(self, force: bool = False, dry_run: bool = False) -> bool:
        """
        Run `prelude update` to refresh inferred context while preserving manual edits.

        Note: `update` does not take a [dir] argument in v1.2.0 — it reads
        from the current context. We run it from repo_path as cwd.

        Args:
            force:   Overwrite all inferred fields.
            dry_run: Preview changes without applying them.
        """
        if not self.cli_available:
            return False

        args = ["update", "--silent"]
        if force:
            args.append("--force")
        if dry_run:
            args.append("--dry-run")

        # update reads from cwd, so we pass cwd here only
        result = self._run(*args, use_cwd=True)
        if result.returncode == 0:
            logger.info("[PRELUDE] Context updated")
            self._cached_export = None  # invalidate cache
            return True
        else:
            logger.warning(f"[PRELUDE] Update failed: {result.stderr}")
            return False

    def refresh(self) -> bool:
        """
        Ensure context is fresh before a GLITCHLAB run.

        If .context/ exists → update it.
        If not → init it.
        Falls back to read-only mode if CLI unavailable but .context/ exists.
        """
        if not self.cli_available:
            if self.context_exists:
                logger.debug("[PRELUDE] Using existing .context/ (CLI not available)")
                return True
            return False

        if self.context_exists:
            self.update()
            return True
        else:
            return self.init()

    # ------------------------------------------------------------------
    # Context Extraction
    # ------------------------------------------------------------------

    def export(self) -> str:
        """
        Get the full Prelude context as markdown.

        Prefers direct file reading (fastest, no subprocess risk).
        Falls back to `prelude export <repo_path> --print --no-copy` if needed.
        """
        if self._cached_export:
            return self._cached_export

        # Prefer direct file reading — deterministic, no hang risk
        if self.context_exists:
            self._cached_export = self._read_context_files()
            if self._cached_export:
                logger.info(f"[PRELUDE] Read context from files ({len(self._cached_export)} chars)")
                return self._cached_export

        # Fallback: CLI export
        if self.cli_available:
            try:
                result = self._run("export", str(self.repo_path), "--print", "--no-copy")
                if result.returncode == 0 and result.stdout.strip():
                    self._cached_export = result.stdout.strip()
                    logger.info(f"[PRELUDE] Exported via CLI ({len(self._cached_export)} chars)")
                    return self._cached_export
                else:
                    logger.warning(f"[PRELUDE] CLI export failed: {result.stderr}")
            except Exception as e:
                logger.warning(f"[PRELUDE] CLI export exception: {e}")

        return ""

    def summary(self) -> dict[str, Any]:
        """
        Get structured summary of project context for logging and status display.
        """
        summary: dict[str, Any] = {
            "available": self.available,
            "cli_installed": self.cli_available,
            "cli_version": self.get_version(),
            "context_dir_exists": self.context_exists,
            "brain_path": str(self.brain_path),
            "files": [],
            "decisions_count": 0,
        }

        if self.context_exists:
            summary["files"] = [
                f.name for f in self.context_dir.iterdir()
                if f.is_file() and not f.name.endswith(".session.json")
            ]

            project_file = self.context_dir / "project.json"
            if project_file.exists():
                try:
                    data = json.loads(project_file.read_text())
                    summary["project_name"] = data.get("name", "unknown")
                    summary["language"] = data.get("language", "unknown")
                    summary["framework"] = data.get("framework", "unknown")
                except (json.JSONDecodeError, KeyError):
                    pass

            stack_file = self.context_dir / "stack.json"
            if stack_file.exists():
                try:
                    data = json.loads(stack_file.read_text())
                    summary["stack"] = data
                except json.JSONDecodeError:
                    pass

            decisions_dir = self.context_dir / "decisions"
            if decisions_dir.is_dir():
                summary["decisions_count"] = len(list(decisions_dir.glob("*.md")))

        return summary

    def get_constraints(self) -> list[str]:
        """Extract project constraints from .context/constraints.json if available."""
        constraints_file = self.context_dir / "constraints.json"
        if not constraints_file.exists():
            return []
        try:
            data = json.loads(constraints_file.read_text())
            if isinstance(data, list):
                return [str(c) for c in data]
            elif isinstance(data, dict):
                constraints = []
                for k, v in data.items():
                    if isinstance(v, str):
                        constraints.append(f"{k}: {v}")
                    elif isinstance(v, list):
                        for item in v:
                            constraints.append(str(item))
                return constraints
        except json.JSONDecodeError:
            pass
        return []

    def get_decisions(self) -> list[str]:
        """Read all Architecture Decision Records as text."""
        decisions = []
        decisions_dir = self.context_dir / "decisions"
        if decisions_dir.is_dir():
            for md_file in sorted(decisions_dir.glob("*.md")):
                decisions.append(md_file.read_text())
        return decisions

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, *args: str, use_cwd: bool = False) -> subprocess.CompletedProcess:
        """
        Run a prelude CLI command.

        Sets PRELUDE_ROOT env var if brain_path is configured.
        Never changes cwd unless use_cwd=True (only needed for `update`).
        """
        env = os.environ.copy()
        if self.brain_path:
            env["PRELUDE_ROOT"] = str(self.brain_path)
            logger.debug(f"[PRELUDE] PRELUDE_ROOT={self.brain_path}")

        cmd = ["prelude", *args]
        logger.debug(f"[PRELUDE] Running: {' '.join(cmd)}")

        return subprocess.run(
            cmd,
            cwd=str(self.repo_path) if use_cwd else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def _read_context_files(self) -> str:
        """Read .context/ files directly and assemble into a single markdown string."""
        parts = []

        # Priority order — most useful context first
        file_order = [
            "project.json",
            "stack.json",
            "architecture.json",
            "architecture.md",
            "constraints.json",
            "changelog.md",
        ]

        seen = set()
        for fname in file_order:
            fpath = self.context_dir / fname
            if fpath.exists():
                seen.add(fname)
                content = fpath.read_text().strip()
                if content:
                    parts.append(f"## {fname}\n\n{content}")

        # ADRs
        decisions_dir = self.context_dir / "decisions"
        if decisions_dir.is_dir():
            for md_file in sorted(decisions_dir.glob("*.md")):
                content = md_file.read_text().strip()
                if content:
                    parts.append(f"## Decision: {md_file.stem}\n\n{content}")

        # Remaining files not yet captured
        for fpath in sorted(self.context_dir.iterdir()):
            if (
                fpath.is_file()
                and fpath.name not in seen
                and not fpath.name.endswith(".session.json")
                and not fpath.suffix == ".session.json"
            ):
                content = fpath.read_text().strip()
                if content:
                    parts.append(f"## {fpath.name}\n\n{content}")

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _version_tuple(version: str) -> tuple[int, ...]:
        """Convert version string to comparable tuple. e.g. '1.2.0' → (1, 2, 0)"""
        try:
            return tuple(int(x) for x in version.strip().split("."))
        except (ValueError, AttributeError):
            return (0, 0, 0)