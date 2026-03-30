from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import yaml

from glitchlab.auditor.task_writer import TaskWriter
from glitchlab.auditor.scanner import ScanResult


class _FakeToolCall:
    def __init__(self, name: str, arguments: dict, tool_id: str = "tool-1"):
        self.id = tool_id
        self.function = SimpleNamespace(name=name, arguments=yaml.safe_dump(arguments, default_flow_style=True))


class _FakeResponse:
    def __init__(self, tool_calls=None, content: str | None = None):
        self.tool_calls = tool_calls or []
        self.content = content


class _FakeRouter:
    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _FakeResponse(
                tool_calls=[
                    _FakeToolCall(
                        "create_task",
                        {
                            "id": "audit-feature-001",
                            "objective": "Create a task file",
                            "constraints": ["keep scope small"],
                            "acceptance": ["task yaml is written"],
                            "risk": "low",
                        },
                    )
                ]
            )
        return _FakeResponse(tool_calls=[_FakeToolCall("done", {"summary": "finished"}, tool_id="tool-2")])


def test_write_tasks_create_task_includes_generated_at_timestamp(tmp_path: Path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    output_dir = tmp_path / "tasks"

    writer = TaskWriter(router=_FakeRouter(), output_dir=output_dir, dry_run=False)
    result = ScanResult(repo_path=repo_path, findings=[])

    written_paths = writer.write_tasks(result)

    assert len(written_paths) == 1
    task_path = written_paths[0]
    data = yaml.safe_load(task_path.read_text())

    assert data["source"] == "auditor"
    assert "generated_at" in data
    parsed = datetime.fromisoformat(data["generated_at"])
    assert parsed.tzinfo is not None
