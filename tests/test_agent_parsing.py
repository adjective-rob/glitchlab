"""Tests for agent parse_response methods.

Only PlannerAgent and TestGenAgent have real parse_response implementations.
The other agents (Implementer, Debugger, Security, Release, Archivist) override
run() with agentic tool loops and have stub parse_response methods.
"""

import json

from glitchlab.agents import AgentContext
from glitchlab.agents.archivist import ArchivistAgent
from glitchlab.agents.debugger import DebuggerAgent
from glitchlab.agents.implementer import ImplementerAgent
from glitchlab.agents.planner import PlannerAgent
from glitchlab.agents.release import ReleaseAgent
from glitchlab.agents.security import SecurityAgent
from glitchlab.agents.testgen import TestGenAgent
from glitchlab.router import RouterResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(content: str) -> RouterResponse:
    """Create a minimal RouterResponse with known fields."""
    return RouterResponse(
        content=content,
        model="test-model-v1",
        tokens_used=42,
        cost=0.001,
    )


def _make_context() -> AgentContext:
    """Create a minimal AgentContext for parse_response calls."""
    return AgentContext(
        task_id="test-task-001",
        objective="Test objective",
        repo_path="/tmp/repo",
        working_dir="/tmp/repo/work",
    )


# ---------------------------------------------------------------------------
# PlannerAgent Tests
# ---------------------------------------------------------------------------

_VALID_PLAN_JSON = json.dumps({
    "steps": [
        {
            "step_number": 1,
            "description": "Modify the config",
            "files": ["config.yaml"],
            "action": "modify",
        }
    ],
    "files_likely_affected": ["config.yaml"],
    "requires_core_change": False,
    "risk_level": "low",
    "risk_notes": "Simple change",
    "test_strategy": ["Run unit tests"],
    "estimated_complexity": "trivial",
    "dependencies_affected": False,
    "public_api_changed": False,
    "self_review_notes": "Looks good",
})


def test_planner_valid_json():
    """PlannerAgent: valid JSON response parses correctly."""
    agent = PlannerAgent(router=None)
    result = agent.parse_response(_make_response(_VALID_PLAN_JSON), _make_context())

    assert "parse_error" not in result
    assert len(result["steps"]) == 1
    assert result["steps"][0]["action"] == "modify"
    assert result["risk_level"] == "low"
    assert result["estimated_complexity"] == "trivial"


def test_planner_json_in_markdown_fences():
    """PlannerAgent: JSON wrapped in markdown code fences still parses."""
    fenced = f"```json\n{_VALID_PLAN_JSON}\n```"
    agent = PlannerAgent(router=None)
    result = agent.parse_response(_make_response(fenced), _make_context())

    assert "parse_error" not in result
    assert len(result["steps"]) == 1


def test_planner_invalid_json_returns_parse_error():
    """PlannerAgent: invalid JSON returns a dict with parse_error=True."""
    agent = PlannerAgent(router=None)
    result = agent.parse_response(
        _make_response("this is not json at all"), _make_context()
    )

    assert result["parse_error"] is True


def test_planner_truncated_json_returns_parse_error():
    """PlannerAgent: truncated/partial JSON returns parse_error=True."""
    truncated = '{"steps": [{"step_number": 1, "description": "incomplete'
    agent = PlannerAgent(router=None)
    result = agent.parse_response(_make_response(truncated), _make_context())

    assert result["parse_error"] is True


def test_planner_successful_parse_includes_meta_keys():
    """PlannerAgent: every successful parse includes _agent, _model, _tokens, _cost."""
    agent = PlannerAgent(router=None)
    result = agent.parse_response(_make_response(_VALID_PLAN_JSON), _make_context())

    assert result["_agent"] == "planner"
    assert result["_model"] == "test-model-v1"
    assert result["_tokens"] == 42
    assert result["_cost"] == 0.001


def test_planner_invalid_action_triggers_validation_error():
    """PlannerAgent: invalid action type triggers Pydantic ValidationError → parse_error."""
    bad_plan = json.dumps({
        "steps": [
            {
                "step_number": 1,
                "description": "Bad action",
                "files": ["foo.py"],
                "action": "refactor",  # Not in Literal["modify", "create", "delete"]
            }
        ],
        "files_likely_affected": ["foo.py"],
        "requires_core_change": False,
        "risk_level": "low",
        "risk_notes": "n/a",
        "test_strategy": [],
        "estimated_complexity": "trivial",
        "dependencies_affected": False,
        "public_api_changed": False,
        "self_review_notes": "",
    })

    agent = PlannerAgent(router=None)
    result = agent.parse_response(_make_response(bad_plan), _make_context())

    assert result["parse_error"] is True


# ---------------------------------------------------------------------------
# TestGenAgent Tests
# ---------------------------------------------------------------------------

_VALID_TESTGEN_JSON = json.dumps({
    "test_file": "tests/test_new.py",
    "content": "def test_it():\n    assert True\n",
    "description": "Basic regression test",
})


def test_testgen_valid_json():
    """TestGenAgent: valid JSON response parses correctly."""
    agent = TestGenAgent(router=None)
    result = agent.parse_response(_make_response(_VALID_TESTGEN_JSON), _make_context())

    assert "parse_error" not in result
    assert result["test_file"] == "tests/test_new.py"
    assert "def test_it" in result["content"]


def test_testgen_json_in_markdown_fences():
    """TestGenAgent: JSON wrapped in markdown code fences still parses."""
    fenced = f"```json\n{_VALID_TESTGEN_JSON}\n```"
    agent = TestGenAgent(router=None)
    result = agent.parse_response(_make_response(fenced), _make_context())

    assert "parse_error" not in result
    assert result["test_file"] == "tests/test_new.py"


def test_testgen_invalid_json_returns_parse_error():
    """TestGenAgent: invalid JSON returns parse_error=True."""
    agent = TestGenAgent(router=None)
    result = agent.parse_response(
        _make_response("not valid json"), _make_context()
    )

    assert result["parse_error"] is True


def test_testgen_truncated_json_returns_parse_error():
    """TestGenAgent: truncated/partial JSON returns parse_error=True."""
    truncated = '{"test_file": "tests/test_x.py", "content": "incomplete'
    agent = TestGenAgent(router=None)
    result = agent.parse_response(_make_response(truncated), _make_context())

    assert result["parse_error"] is True


def test_testgen_successful_parse_includes_meta_keys():
    """TestGenAgent: every successful parse includes _agent, _model, _tokens, _cost."""
    agent = TestGenAgent(router=None)
    result = agent.parse_response(_make_response(_VALID_TESTGEN_JSON), _make_context())

    assert result["_agent"] == "testgen"
    assert result["_model"] == "test-model-v1"
    assert result["_tokens"] == 42
    assert result["_cost"] == 0.001


def test_testgen_missing_test_file_triggers_parse_error():
    """TestGenAgent: missing test_file field triggers parse_error."""
    missing_field = json.dumps({
        "content": "def test_it(): pass",
        "description": "A test",
    })

    agent = TestGenAgent(router=None)
    result = agent.parse_response(_make_response(missing_field), _make_context())

    assert result["parse_error"] is True


# ---------------------------------------------------------------------------
# Stub Agent Tests (parse_response is unused in tool-loop agents)
# ---------------------------------------------------------------------------

def test_implementer_parse_response_is_stub():
    """ImplementerAgent: parse_response is a no-op stub (returns None)."""
    agent = ImplementerAgent(router=None)
    result = agent.parse_response(_make_response("{}"), _make_context())
    assert result is None


def test_debugger_parse_response_is_stub():
    """DebuggerAgent: parse_response is a no-op stub (returns None)."""
    agent = DebuggerAgent(router=None)
    result = agent.parse_response(_make_response("{}"), _make_context())
    assert result is None


def test_security_parse_response_is_stub():
    """SecurityAgent: parse_response is a no-op stub (returns None)."""
    agent = SecurityAgent(router=None)
    result = agent.parse_response(_make_response("{}"), _make_context())
    assert result is None


def test_release_parse_response_is_stub():
    """ReleaseAgent: parse_response returns empty dict (stub)."""
    agent = ReleaseAgent(router=None)
    result = agent.parse_response(_make_response("{}"), _make_context())
    assert result == {}


def test_archivist_parse_response_is_stub():
    """ArchivistAgent: parse_response is a no-op stub (returns None)."""
    agent = ArchivistAgent(router=None)
    result = agent.parse_response(_make_response("{}"), _make_context())
    assert result is None
