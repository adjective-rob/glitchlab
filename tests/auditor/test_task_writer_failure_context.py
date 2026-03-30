from pathlib import Path
from types import SimpleNamespace

from glitchlab.auditor.task_writer import TaskWriter
from glitchlab.auditor.scanner import ScanResult
from glitchlab.history import TaskHistory


class _FakeToolCall:
    """Tool call object that supports both model_dump() and attribute access."""

    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)

    def model_dump(self):
        return {
            "id": self.id,
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


class _FakeRouter:
    def __init__(self):
        self.calls = 0
        self.captured_messages = []

    def complete(self, **kwargs):
        self.calls += 1
        self.captured_messages = kwargs.get("messages", [])
        return SimpleNamespace(
            content=None,
            tool_calls=[
                _FakeToolCall("tc-1", "done", '{"summary": "finished"}'),
            ],
        )


def test_gather_failure_context_empty_when_no_history(tmp_path: Path):
    writer = TaskWriter(router=_FakeRouter(), output_dir=tmp_path, dry_run=True)
    ctx = writer._gather_failure_context(tmp_path)
    assert ctx == ""


def test_gather_failure_context_returns_formatted_failures(tmp_path: Path):
    history = TaskHistory(tmp_path)
    history.record({"task_id": "task-001", "status": "failed", "error": "tests broke"})
    history.record({"task_id": "task-002", "status": "pr_created", "error": None})
    history.record({"task_id": "task-003", "status": "budget_exceeded", "error": "over limit"})

    writer = TaskWriter(router=_FakeRouter(), output_dir=tmp_path, dry_run=True)
    ctx = writer._gather_failure_context(tmp_path)

    assert "Previous Task Failures" in ctx
    assert "task-001" in ctx
    assert "tests broke" in ctx
    assert "task-003" in ctx
    assert "over limit" in ctx
    # Successful task should not appear
    assert "task-002" not in ctx


def test_failure_context_injected_into_user_prompt(tmp_path: Path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    history = TaskHistory(repo_path)
    history.record({"task_id": "broken-task", "status": "failed", "error": "syntax error"})

    router = _FakeRouter()
    writer = TaskWriter(router=router, output_dir=tmp_path / "tasks", dry_run=True)
    result = ScanResult(repo_path=repo_path, findings=[])

    writer.write_tasks(result)

    user_msg = router.captured_messages[1]["content"]
    assert "Previous Task Failures" in user_msg
    assert "broken-task" in user_msg
    assert "syntax error" in user_msg
