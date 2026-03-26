from pathlib import Path
from types import SimpleNamespace

from glitchlab.auditor.task_writer import TaskWriter
from glitchlab.auditor.scanner import ScanResult
from glitchlab.config_loader import load_config
from glitchlab.router import Router


def test_task_writer_routes_with_auditor_role_and_router_resolves_auditor_model(tmp_path: Path):
    config = load_config()
    router = Router(config)

    captured = {}

    def fake_complete(*, role, messages, tools, **kwargs):
        captured["role"] = role
        captured["messages"] = messages
        captured["tools"] = tools
        captured["kwargs"] = kwargs
        tool_call = SimpleNamespace(
            id="tc-1",
            function=SimpleNamespace(name="done", arguments='{"summary": "finished"}')
        )
        return SimpleNamespace(content=None, tool_calls=[tool_call])

    router.complete = fake_complete

    writer = TaskWriter(router=router, output_dir=tmp_path, dry_run=True)
    result = ScanResult(repo_path=tmp_path, findings=[])

    written = writer.write_tasks(result)

    assert written == []
    assert captured["role"] == "auditor"
    assert captured["kwargs"]["tool_choice"] == {"type": "function", "function": {"name": "think"}}
    assert router.resolve_model("auditor") == "openai/gpt-5.4-mini"
