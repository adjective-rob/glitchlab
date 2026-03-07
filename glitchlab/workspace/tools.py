"""
GLITCHLAB Tool Execution Layer

Agents do NOT run arbitrary commands. The controller exposes
a constrained set of safe tools. Everything else is blocked.
"""

import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from glitchlab.event_bus import bus


@dataclass
class ToolResult:
    command: str
    stdout: str
    stderr: str
    returncode: int
    allowed: bool = True
    action_id: Optional[str] = None  # Added for traceability across the engine

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

    def execute(
        self, 
        command: str, 
        timeout: int = 120,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> ToolResult:
        """
        Execute a command if it passes safety checks.

        Args:
            command: Shell command string
            timeout: Max seconds before kill
            run_id: Execution session ID for Zephyr tracking
            agent_id: The identifier of the agent invoking the tool

        Returns:
            ToolResult with output and status
        """
        action_id = f"act-{uuid.uuid4()}"
        agent_identity = agent_id or "system"

        # Emit intent to execute (Zephyr can hash the command intent here)
        bus.emit(
            event_type="action.started",
            payload={"command": command, "working_dir": str(self.working_dir)},
            agent_id=agent_identity,
            run_id=run_id,
            action_id=action_id
        )

        # Check blocked patterns first
        for pattern in self.blocked_patterns:
            if pattern in command:
                result = ToolResult(
                    command=command,
                    stdout="",
                    stderr=f"BLOCKED: Command contains forbidden pattern: {pattern}",
                    returncode=-1,
                    allowed=False,
                    action_id=action_id
                )
                self._execution_log.append(result)
                logger.warning(f"[TOOLS] BLOCKED: {command} (pattern: {pattern})")
                self._emit_completion(result, run_id, agent_identity, action_id)
                raise ToolViolationError(f"Blocked pattern detected: {pattern}")

        # Check allowlist
        if not self._is_allowed(command):
            result = ToolResult(
                command=command,
                stdout="",
                stderr=f"DENIED: Command not in allowlist. Allowed: {self.allowed_tools}",
                returncode=-1,
                allowed=False,
                action_id=action_id
            )
            self._execution_log.append(result)
            logger.warning(f"[TOOLS] DENIED: {command}")
            self._emit_completion(result, run_id, agent_identity, action_id)
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
                action_id=action_id
            )
        except subprocess.TimeoutExpired:
            result = ToolResult(
                command=command,
                stdout="",
                stderr=f"TIMEOUT: Command exceeded {timeout}s",
                returncode=-1,
                action_id=action_id
            )

        self._execution_log.append(result)

        # Emit completion for Zephyr to sign the outcome
        self._emit_completion(result, run_id, agent_identity, action_id)

        if result.success:
            logger.debug(f"[TOOLS] OK: {command}")
        else:
            logger.warning(f"[TOOLS] FAIL ({result.returncode}): {command}")

        return result

    def _emit_completion(
        self, 
        result: ToolResult, 
        run_id: Optional[str], 
        agent_id: str, 
        action_id: str
    ) -> None:
        """Helper to broadcast the result to the EventBus so Zephyr can sign it."""
        bus.emit(
            event_type="action.completed",
            payload={
                "command": result.command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "allowed": result.allowed
            },
            agent_id=agent_id,
            run_id=run_id,
            action_id=action_id
        )

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