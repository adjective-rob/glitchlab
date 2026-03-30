"""Tests for AgentResult model."""

from glitchlab.agents import AgentResult


def test_from_raw_successful_planner_dict():
    """from_raw preserves planner metadata for a successful payload."""
    raw = {
        "plan": ["step1", "step2"],
        "reasoning": "looks good",
        "_agent": "planner",
        "_model": "gpt-4",
        "_tokens": 150,
        "_cost": 0.01,
    }
    result = AgentResult.from_raw(raw)
    assert result.status == "success"
    assert result.agent == "planner"
    assert result.model == "gpt-4"
    assert result.tokens_used == 150
    assert result.cost == 0.01


def test_from_raw_parse_error_produces_error_status():
    """from_raw marks parse failures as error results."""
    raw = {
        "parse_error": True,
        "raw_response": "malformed output",
        "_agent": "implementer",
    }
    result = AgentResult.from_raw(raw)
    assert result.status == "error"
    assert result.error == "malformed output"


def test_from_raw_delegating_status():
    """from_raw preserves an explicit delegating status."""
    raw = {
        "_status": "delegating",
        "_agent": "planner",
        "target": "implementer",
    }
    result = AgentResult.from_raw(raw)
    assert result.status == "delegating"


def test_payload_excludes_underscore_prefixed_keys():
    """from_raw excludes internal underscore-prefixed keys from payload."""
    raw = {
        "plan": ["step1"],
        "reasoning": "ok",
        "_agent": "planner",
        "_model": "gpt-4",
        "_tokens": 100,
        "_cost": 0.005,
        "_status": "success",
    }
    result = AgentResult.from_raw(raw)
    assert "_agent" not in result.payload
    assert "_model" not in result.payload
    assert "_tokens" not in result.payload
    assert "_cost" not in result.payload
    assert "_status" not in result.payload
    assert "plan" in result.payload
    assert "reasoning" in result.payload


def test_default_construction():
    """AgentResult defaults to an empty successful result."""
    result = AgentResult()
    assert result.status == "success"
    assert result.payload == {}
    assert result.error is None
    assert result.agent == ""
    assert result.model == ""
    assert result.tokens_used == 0
    assert result.cost == 0.0
