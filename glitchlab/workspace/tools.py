"""
GLITCHLAB Tool Execution Layer

Agents do NOT run arbitrary commands. The controller exposes
a constrained set of safe tools. Everything else is blocked.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class ToolResult:
    command: str
    stdout: str
    stderr: str
    returncode: int
    allowed: bool = True

    @property
    def success(self) -> bool:
        return self.returncode == 0 and self.allowed


class ToolViolationError(Exception):
    pass


class ToolExecutor:
    """
    Sandboxed command executor.

    Only runs commands that match the allowlist.
    Rejects anything matching blocked patterns.
    All execution is scoped to a working directory.
    """

    def __init__(
        self,
        allowed_tools: list[str],
        blocked_patterns: list[str],
        working_dir: Path,
    ) -> None:
        self.allowed_tools = allowed_tools
        self.blocked_patterns = blocked_patterns
        self.working_dir = working_dir
        self._execution_log: list[ToolResult] = []

    def execute(self, command: str, timeout: int = 120) -> ToolResult:
        """
        Execute a command if it passes safety checks.

        Args:
            command: Shell command string
            timeout: Max seconds before kill

        Returns:
            ToolResult with output and status
        """
        # Check blocked patterns first
        for pattern in self.blocked_patterns:
            if pattern in command:
                result = ToolResult(
                    command=command,
                    stdout="",
                    stderr=f"BLOCKED: Command contains forbidden pattern: {pattern}",
                    returncode=-1,
                    allowed=False,
                )
                self._execution_log.append(result)
                logger.warning(f"[TOOLS] BLOCKED: {command} (pattern: {pattern})")
                raise ToolViolationError(f"Blocked pattern detected: {pattern}")

        # Check allowlist
        if not self._is_allowed(command):
            result = ToolResult(
                command=command,
                stdout="",
                stderr=f"DENIED: Command not in allowlist. Allowed: {self.allowed_tools}",
                returncode=-1,
                allowed=False,
            )
            self._execution_log.append(result)
            logger.warning(f"[TOOLS] DENIED: {command}")
            raise ToolViolationError(f"Command not allowed: {command}")

        # Execute
        logger.info(f"[TOOLS] Running: {command}")
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result = ToolResult(
                command=command,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            result = ToolResult(
                command=command,
                stdout="",
                stderr=f"TIMEOUT: Command exceeded {timeout}s",
                returncode=-1,
            )

        self._execution_log.append(result)

        if result.success:
            logger.debug(f"[TOOLS] OK: {command}")
        else:
            logger.warning(f"[TOOLS] FAIL ({result.returncode}): {command}")

        return result

    def _is_allowed(self, command: str) -> bool:
        """Check if command matches any allowlist entry (prefix match)."""
        cmd_stripped = command.strip()
        for allowed in self.allowed_tools:
            if cmd_stripped.startswith(allowed):
                return True
        return False

    @property
    def execution_log(self) -> list[ToolResult]:
        return list(self._execution_log)

    def clear_log(self) -> None:
        self._execution_log.clear()
