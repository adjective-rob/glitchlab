from pathlib import Path
from types import SimpleNamespace

from glitchlab.agents.implementer import ImplementerAgent


class DummyRouter:
    def __init__(self):
        self.calls = 0

    def complete(self, role, messages, tools, **kwargs):
        self.calls += 1
        if self.calls == 1:
            tool_call = SimpleNamespace(
                id="tc-think",
                function=SimpleNamespace(
                    name="think",
                    arguments='{"search_strategy":"inspect target","execution_plan":"run verification"}',
                ),
            )
            return SimpleNamespace(
                content=None,
                tool_calls=[tool_call],
                tokens_used=1,
                model="test-model",
                cost=0,
            )

        tool_call = SimpleNamespace(
            id="tc-check",
            function=SimpleNamespace(
                name="run_check",
                arguments='{"command":"python -V"}',
            ),
        )
        return SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
            tokens_used=1,
            model="test-model",
            cost=0,
        )


class DummyToolExecutor:
    def execute(self, command, run_id, agent_id):
        return SimpleNamespace(returncode=0, stdout="all good", stderr="")


class TestImplementerAgent(ImplementerAgent):
    def __init__(self, router):
        self.router = router


def test_run_check_success_appends_done_prompt(tmp_path: Path):
    agent = TestImplementerAgent(DummyRouter())
    context = SimpleNamespace(
        previous_output={"plan_steps": []},
        file_context={},
        extra={"tool_executor": DummyToolExecutor(), "fast_mode": True},
        repo_path=str(tmp_path),
        working_dir=str(tmp_path),
        objective="Verify changes",
        run_id="run-123",
    )

    result = agent.run(context)

    assert result["parse_error"] is True
    run_check_messages = [
        m for m in result["_messages"] if m.get("role") == "tool" and m.get("name") == "run_check"
    ]
    assert run_check_messages
    assert run_check_messages[-1]["content"].endswith(
        "Verification passed. If your changes are complete, call done now."
    )
