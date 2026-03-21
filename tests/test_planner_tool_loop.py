"""Tests for the planner's agentic tool loop."""
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

from glitchlab.agents import AgentContext
from glitchlab.agents.planner import PlannerAgent, PLANNER_TOOLS
from glitchlab.router import RouterResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(tmp_path: Path, symbol_index=None) -> AgentContext:
    return AgentContext(
        task_id="test-task-001",
        objective="Add a health check endpoint",
        repo_path=str(tmp_path),
        working_dir=str(tmp_path),
        extra={"symbol_index": symbol_index} if symbol_index else {},
    )


def _make_tool_call(tc_id: str, name: str, arguments: dict) -> MagicMock:
    """Create a mock tool_call matching the litellm interface."""
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    tc.model_dump.return_value = {
        "id": tc_id,
        "function": {"name": name, "arguments": json.dumps(arguments)},
        "type": "function",
    }
    return tc


def _make_response(content: str = None, tool_calls: list = None, model: str = "test-model") -> RouterResponse:
    return RouterResponse(
        content=content,
        model=model,
        tokens_used=100,
        cost=0.01,
        latency_ms=500,
        tool_calls=tool_calls,
    )


_VALID_PLAN = json.dumps({
    "steps": [
        {
            "step_number": 1,
            "description": "Add health check route",
            "files": ["app/routes.py"],
            "action": "modify",
        }
    ],
    "files_likely_affected": ["app/routes.py"],
    "requires_core_change": False,
    "risk_level": "low",
    "risk_notes": "Simple addition",
    "test_strategy": ["Run unit tests"],
    "estimated_complexity": "trivial",
    "dependencies_affected": False,
    "public_api_changed": False,
    "self_review_notes": "Looks good",
})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_planner_calls_think_then_submit(tmp_path):
    """Planner: think on step 0, then submit_plan on step 1 → valid plan returned."""
    mock_router = MagicMock()

    # Step 0: forced think
    think_tc = _make_tool_call("tc_1", "think", {"analysis": "Simple task, one file change"})
    # Step 1: submit_plan
    submit_tc = _make_tool_call("tc_2", "submit_plan", {"plan_json": _VALID_PLAN})

    mock_router.complete.side_effect = [
        _make_response(tool_calls=[think_tc]),
        _make_response(tool_calls=[submit_tc]),
    ]

    agent = PlannerAgent(router=mock_router)
    ctx = _make_context(tmp_path)
    result = agent.run(ctx)

    assert "parse_error" not in result
    assert len(result["steps"]) == 1
    assert result["risk_level"] == "low"
    assert mock_router.complete.call_count == 2


def test_planner_uses_get_function(tmp_path):
    """Planner: get_function tool call queries symbol_index and returns body."""
    mock_router = MagicMock()
    mock_symbol_index = MagicMock()
    mock_symbol_index.get_function_body.return_value = {
        "file": "app/routes.py",
        "line_start": 10,
        "line_end": 20,
        "body": "def health():\n    return {'status': 'ok'}",
    }

    # Step 0: think
    think_tc = _make_tool_call("tc_1", "think", {"analysis": "Need to read the existing route"})
    # Step 1: get_function
    get_fn_tc = _make_tool_call("tc_2", "get_function", {"symbol": "health", "file": "app/routes.py"})
    # Step 2: submit_plan
    submit_tc = _make_tool_call("tc_3", "submit_plan", {"plan_json": _VALID_PLAN})

    mock_router.complete.side_effect = [
        _make_response(tool_calls=[think_tc]),
        _make_response(tool_calls=[get_fn_tc]),
        _make_response(tool_calls=[submit_tc]),
    ]

    agent = PlannerAgent(router=mock_router)
    ctx = _make_context(tmp_path, symbol_index=mock_symbol_index)
    result = agent.run(ctx)

    assert "parse_error" not in result
    mock_symbol_index.get_function_body.assert_called_once_with("health", "app/routes.py")


def test_planner_loop_exhaustion_returns_parse_error(tmp_path):
    """Planner: if loop exhausts without submit_plan, return parse_error."""
    mock_router = MagicMock()

    # Return think calls for all 8 steps — never submit
    think_tc = _make_tool_call("tc_1", "think", {"analysis": "Still thinking..."})
    mock_router.complete.return_value = _make_response(tool_calls=[think_tc])

    agent = PlannerAgent(router=mock_router)
    ctx = _make_context(tmp_path)
    result = agent.run(ctx)

    assert result["parse_error"] is True
    assert result["risk_level"] == "high"
    assert mock_router.complete.call_count == 8


def test_planner_plain_text_fallback(tmp_path):
    """Planner: if model returns plain text JSON (no tool calls), parse_response handles it."""
    mock_router = MagicMock()

    # Step 0: forced think
    think_tc = _make_tool_call("tc_1", "think", {"analysis": "Simple"})
    # Step 1: plain text JSON response (no tool calls)
    mock_router.complete.side_effect = [
        _make_response(tool_calls=[think_tc]),
        _make_response(content=_VALID_PLAN, tool_calls=None),
    ]

    agent = PlannerAgent(router=mock_router)
    ctx = _make_context(tmp_path)
    result = agent.run(ctx)

    assert "parse_error" not in result
    assert len(result["steps"]) == 1
