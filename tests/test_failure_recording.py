"""Tests for failure recording and context injection via TaskHistory."""
import json
from pathlib import Path

from glitchlab.history import TaskHistory


def test_failure_is_recorded_to_jsonl(tmp_path: Path):
    """A failed task result is persisted to the history JSONL file."""
    history = TaskHistory(tmp_path)

    history.record({
        "task_id": "task_123",
        "status": "implementation_failed",
        "error": "Could not find target file",
    })

    assert history.history_file.exists()
    lines = history.history_file.read_text().strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["task_id"] == "task_123"
    assert entry["status"] == "implementation_failed"


def test_failure_context_includes_recent_failures(tmp_path: Path):
    """build_failure_context returns a string mentioning recent failed tasks."""
    history = TaskHistory(tmp_path)

    # Record a failure
    history.record({
        "task_id": "task_200",
        "status": "error",
        "error": "ModuleNotFoundError: No module named 'foo'",
    })

    context = history.build_failure_context()

    # Should be non-empty and reference the failure
    assert context
    assert "task_200" in context or "error" in context.lower()


def test_no_failure_context_when_all_succeeded(tmp_path: Path):
    """build_failure_context returns empty string when no recent failures."""
    history = TaskHistory(tmp_path)

    history.record({"task_id": "task_300", "status": "pr_created"})
    history.record({"task_id": "task_301", "status": "committed"})

    context = history.build_failure_context()
    assert not context


def test_multiple_failures_all_appear_in_context(tmp_path: Path):
    """build_failure_context includes all recent failures, not just the last."""
    history = TaskHistory(tmp_path)

    history.record({"task_id": "task_400", "status": "error", "error": "SyntaxError in config.py"})
    history.record({"task_id": "task_401", "status": "plan_failed", "error": "Planner timeout"})

    context = history.build_failure_context()
    assert "task_400" in context
    assert "task_401" in context
