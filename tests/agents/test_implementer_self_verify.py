"""Tests for implementer self-verify gate: done is rejected if writes happened without run_check."""

from pathlib import Path
from types import SimpleNamespace

from glitchlab.agents.implementer import ImplementerAgent


class DummyToolExecutor:
    def execute(self, command, run_id, agent_id):
        return SimpleNamespace(returncode=0, stdout="all good", stderr="")


class _ToolCall(dict):
    """Dict-like tool call mock that also supports attribute access (like litellm objects)."""

    def __init__(self, id, name, arguments):
        super().__init__(id=id, function={"name": name, "arguments": arguments})
        self.id = id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class TestImplementerAgent(ImplementerAgent):
    def __init__(self, router):
        self.router = router


class RouterDoneRejected:
    """Simulates: think -> write_file -> done (rejected) -> run_check -> done (accepted)."""

    def __init__(self, tmp_path):
        self.calls = 0
        self.tmp_path = tmp_path

    def complete(self, role, messages, tools, **kwargs):
        self.calls += 1

        if self.calls == 1:
            return SimpleNamespace(
                content=None,
                tool_calls=[_ToolCall(
                    "tc-think", "think",
                    '{"search_strategy":"plan","execution_plan":"write and verify"}',
                )],
                tokens_used=1, model="test-model", cost=0,
            )

        if self.calls == 2:
            return SimpleNamespace(
                content=None,
                tool_calls=[_ToolCall(
                    "tc-write", "write_file",
                    '{"path":"hello.py","content":"print(1)"}',
                )],
                tokens_used=1, model="test-model", cost=0,
            )

        if self.calls == 3:
            return SimpleNamespace(
                content=None,
                tool_calls=[_ToolCall(
                    "tc-done-early", "done",
                    '{"summary":"finished"}',
                )],
                tokens_used=1, model="test-model", cost=0,
            )

        if self.calls == 4:
            return SimpleNamespace(
                content=None,
                tool_calls=[_ToolCall(
                    "tc-check", "run_check",
                    '{"command":"python -V"}',
                )],
                tokens_used=1, model="test-model", cost=0,
            )

        return SimpleNamespace(
            content=None,
            tool_calls=[_ToolCall(
                "tc-done-final", "done",
                '{"summary":"finished after verify"}',
            )],
            tokens_used=1, model="test-model", cost=0,
        )


def test_done_rejected_without_run_check(tmp_path: Path):
    """done is rejected when writes happened without a passing run_check."""
    agent = TestImplementerAgent(RouterDoneRejected(tmp_path))
    context = SimpleNamespace(
        previous_output={"plan_steps": []},
        file_context={},
        extra={"tool_executor": DummyToolExecutor(), "fast_mode": True},
        repo_path=str(tmp_path),
        working_dir=str(tmp_path),
        objective="Write and verify",
        run_id="run-self-verify",
    )

    result = agent.run(context)

    # The first done call should have been rejected
    done_messages = [
        m for m in result["_messages"]
        if m.get("role") == "tool" and m.get("name") == "done"
    ]
    assert len(done_messages) == 1  # the rejection message only

    rejected = done_messages[0]
    assert "have not run any verification" in rejected["content"]
    assert rejected["tool_call_id"] == "tc-done-early"

    # The second done succeeded (returned the result dict)
    assert result["summary"] == "finished after verify"


class RouterDoneWithoutWrites:
    """Simulates: think -> done. No writes, so done should succeed immediately."""

    def __init__(self):
        self.calls = 0

    def complete(self, role, messages, tools, **kwargs):
        self.calls += 1

        if self.calls == 1:
            return SimpleNamespace(
                content=None,
                tool_calls=[_ToolCall(
                    "tc-think", "think",
                    '{"search_strategy":"plan","execution_plan":"nothing to write"}',
                )],
                tokens_used=1, model="test-model", cost=0,
            )

        return SimpleNamespace(
            content=None,
            tool_calls=[_ToolCall(
                "tc-done", "done",
                '{"summary":"no changes needed"}',
            )],
            tokens_used=1, model="test-model", cost=0,
        )


def test_done_allowed_without_writes(tmp_path: Path):
    """done succeeds immediately when no writes have been made."""
    agent = TestImplementerAgent(RouterDoneWithoutWrites())
    context = SimpleNamespace(
        previous_output={"plan_steps": []},
        file_context={},
        extra={"tool_executor": DummyToolExecutor(), "fast_mode": True},
        repo_path=str(tmp_path),
        working_dir=str(tmp_path),
        objective="Check only",
        run_id="run-no-writes",
    )

    result = agent.run(context)
    assert result["summary"] == "no changes needed"
