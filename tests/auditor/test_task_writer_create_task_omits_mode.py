from pathlib import Path
from types import SimpleNamespace

import yaml

from glitchlab.auditor.task_writer import TaskWriter
from glitchlab.auditor.scanner import ScanResult


class _FakeToolFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _FakeToolFunction(name, arguments)


class _FakeResponse:
    def __init__(self, tool_calls, content=None):
        self.tool_calls = tool_calls
        self.content = content


class _FakeRouter:
    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _FakeResponse(
                [
                    _FakeToolCall(
                        "call-1",
                        "create_task",
                        '{"id":"audit-feature-001","objective":"Create a valid task","constraints":[],"acceptance":["task file is written"],"risk":"low"}',
                    )
                ]
            )
        return _FakeResponse([
            _FakeToolCall("call-2", "done", '{"summary":"finished"}')
        ])


def test_task_writer_writes_auditor_task_without_mode_field(tmp_path: Path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    output_dir = tmp_path / "tasks"

    writer = TaskWriter(router=_FakeRouter(), output_dir=output_dir, dry_run=False)
    result = ScanResult(repo_path=repo_path, findings=[])

    written = writer.write_tasks(result)

    assert len(written) == 1
    task_path = written[0]
    data = yaml.safe_load(task_path.read_text())

    assert data["id"] == "audit-feature-001"
    assert data["source"] == "auditor"
    assert "mode" not in data
