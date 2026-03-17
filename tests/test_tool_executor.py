"""Tests for ToolExecutor from glitchlab/workspace/tools.py."""

import pytest

from glitchlab.workspace.tools import ToolExecutor, ToolViolationError


@pytest.fixture
def executor(tmp_path):
    """Create a ToolExecutor with a basic allowlist and blocklist."""
    return ToolExecutor(
        allowed_tools=["ls", "echo", "cat", "sleep"],
        blocked_patterns=["rm -rf", "&&", "||"],
        working_dir=tmp_path,
    )


def test_allowed_command_succeeds(executor):
    """An allowed command executes and returns ToolResult with success=True."""
    result = executor.execute("echo hello")

    assert result.success is True
    assert result.returncode == 0
    assert "hello" in result.stdout


def test_command_not_in_allowlist_raises(tmp_path):
    """A command not in the allowlist raises ToolViolationError."""
    executor = ToolExecutor(
        allowed_tools=["ls"],
        blocked_patterns=[],
        working_dir=tmp_path,
    )

    with pytest.raises(ToolViolationError):
        executor.execute("curl http://example.com")


def test_blocked_pattern_raises(executor):
    """A command containing a blocked pattern raises ToolViolationError."""
    with pytest.raises(ToolViolationError):
        executor.execute("rm -rf /tmp/something")


def test_timeout_returns_failure(executor):
    """A command that times out returns returncode=-1 and TIMEOUT in stderr."""
    result = executor.execute("sleep 5", timeout=1)

    assert result.returncode == -1
    assert "TIMEOUT" in result.stderr


def test_execution_log_records_commands(executor):
    """The execution_log records all attempted commands."""
    executor.execute("echo first")
    executor.execute("echo second")

    log = executor.execution_log
    assert len(log) == 2
    assert log[0].command == "echo first"
    assert log[1].command == "echo second"


def test_clear_log_empties_execution_log(executor):
    """clear_log empties the execution log."""
    executor.execute("echo test")
    assert len(executor.execution_log) == 1

    executor.clear_log()
    assert len(executor.execution_log) == 0


def test_prefix_matching_allows_extended_command(tmp_path):
    """Prefix matching works — 'cargo test' allows 'cargo test --release'."""
    executor = ToolExecutor(
        allowed_tools=["cargo test"],
        blocked_patterns=[],
        working_dir=tmp_path,
    )

    # cargo test isn't installed, but the allowlist check should pass
    # and we should get a subprocess error, not a ToolViolationError
    result = executor.execute("cargo test --release")

    # The command was allowed (not blocked by allowlist)
    assert result.allowed is True
    # It will fail because cargo isn't installed, but that's fine —
    # the point is it wasn't rejected by the allowlist
    assert result.returncode != 0


def test_blocked_pattern_takes_priority_over_allowlist(tmp_path):
    """A command matching both blocked and allowed is blocked."""
    executor = ToolExecutor(
        allowed_tools=["rm"],
        blocked_patterns=["rm -rf"],
        working_dir=tmp_path,
    )

    with pytest.raises(ToolViolationError):
        executor.execute("rm -rf foo")
