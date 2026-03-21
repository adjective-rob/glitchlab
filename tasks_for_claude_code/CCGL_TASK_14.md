# Task 14: Fix stale `test_failure_recording.py`

**File:** `tests/test_failure_recording.py`

**Problem:** This test was written for an older Controller API. It references `Controller(workspace_dir=...)`, `controller._execute_task`, `controller.run_task`, and `glitchlab.controller.Planner` — none of which exist. The actual `Controller.__init__` takes `(repo_path: Path, config=None, allow_core=False, auto_approve=False, test_command=None)`.

**What to do:** Rewrite the test to exercise the actual failure recording path using `TaskHistory` directly — that's the real unit under test. Do NOT instantiate `Controller` (it requires a real git repo, Router, agents, etc).

**Reference:** `tests/test_history.py` for patterns, `glitchlab/history.py` for the `TaskHistory` API. The key methods are `record()`, `get_failures()`, and `build_failure_context()`.

**Replace the entire file with:**

```python
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
    history.record({"task_id": "task_301", "status": "merged"})

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
```

### Do NOT touch

- Any other test file.

### Verify

```bash
python -m pytest tests/test_failure_recording.py -v
python -m pytest tests/ -x
```
