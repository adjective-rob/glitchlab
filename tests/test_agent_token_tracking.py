import pytest
from unittest.mock import MagicMock, patch
import json
from glitchlab.agents.debugger import DebuggerAgent
from glitchlab.agents.implementer import ImplementerAgent
from glitchlab.agents import AgentContext
from glitchlab.router import RouterResponse

class MockToolCall:
    def __init__(self, name, arguments):
        self.id = "call_123"
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments

    def model_dump(self):
        return {"id": self.id, "function": {"name": self.function.name, "arguments": self.function.arguments}}

@pytest.fixture
def mock_router():
    router = MagicMock()
    # First call: think, Second call: done
    router.complete.side_effect = [
        RouterResponse(
            content="",
            tool_calls=[MockToolCall("think", json.dumps({"hypothesis": "h", "investigation_plan": "p", "search_strategy": "s", "execution_plan": "e"}))],
            tokens_used=100,
            cost=0.01,
            model="test-model"
        ),
        RouterResponse(
            content="",
            tool_calls=[MockToolCall("done", json.dumps({"diagnosis": "d", "root_cause": "r", "fix_summary": "f", "confidence": "high", "commit_message": "c", "summary": "s"}))],
            tokens_used=150,
            cost=0.015,
            model="test-model"
        )
    ]
    return router

@pytest.fixture
def mock_event_bus():
    bus = MagicMock()
    return bus

def test_debugger_token_tracking(mock_router, mock_event_bus):
    agent = DebuggerAgent(router=mock_router)
    context = AgentContext(
        task_id="test_task",
        repo_path=".",
        objective="Fix bug",
        working_dir=".",
        previous_output={}, 
        extra={"event_bus": mock_event_bus, "test_command": "pytest"}
    )
    
    result = agent.run(context)
    
    assert result["_loop_tokens"] == 250
    
    loop_step_events = [call for call in mock_event_bus.emit.call_args_list if call.kwargs.get("event_type") == "agent.loop_step"]
    assert len(loop_step_events) == 2
    assert loop_step_events[0].kwargs["payload"]["step_tokens"] == 100
    assert loop_step_events[0].kwargs["payload"]["cumulative_tokens"] == 100
    assert loop_step_events[1].kwargs["payload"]["step_tokens"] == 150
    assert loop_step_events[1].kwargs["payload"]["cumulative_tokens"] == 250

@patch("glitchlab.agents.implementer.bus")
def test_implementer_token_tracking(mock_global_bus, mock_router):
    agent = ImplementerAgent(router=mock_router)
    context = AgentContext(
        task_id="test_task",
        repo_path=".",
        objective="Implement feature",
        working_dir=".",
        previous_output={}, 
        extra={}
    )
    
    result = agent.run(context)
    
    assert result["_loop_tokens"] == 250
    
    loop_step_events = [call for call in mock_global_bus.emit.call_args_list if call.kwargs.get("event_type") == "agent.loop_step"]
    assert len(loop_step_events) == 2
    assert loop_step_events[0].kwargs["payload"]["step_tokens"] == 100
    assert loop_step_events[0].kwargs["payload"]["cumulative_tokens"] == 100
    assert loop_step_events[1].kwargs["payload"]["step_tokens"] == 150
    assert loop_step_events[1].kwargs["payload"]["cumulative_tokens"] == 250
