"""Tests for YAML agent messaging format."""

import yaml
from glitchlab.agents import BaseAgent, AgentContext
from glitchlab.router import RouterResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubRouter:
    """Minimal router stub for testing BaseAgent._yaml_block."""
    pass


class _ConcreteAgent(BaseAgent):
    """Concrete subclass so we can instantiate BaseAgent."""
    role = "test"
    system_prompt = "test"

    def build_messages(self, context):
        return []

    def parse_response(self, response, context):
        return {}


def _make_agent() -> _ConcreteAgent:
    return _ConcreteAgent(router=_StubRouter())


# ---------------------------------------------------------------------------
# _yaml_block
# ---------------------------------------------------------------------------

def test_yaml_block_produces_valid_yaml():
    agent = _make_agent()
    data = {"task": "fix bug", "files": ["a.py", "b.py"], "count": 3}
    result = agent._yaml_block(data)
    parsed = yaml.safe_load(result)
    assert parsed == data


def test_yaml_block_preserves_key_order():
    agent = _make_agent()
    data = {"z_last": 1, "a_first": 2, "m_middle": 3}
    result = agent._yaml_block(data)
    keys = [line.split(":")[0] for line in result.strip().splitlines()]
    assert keys == ["z_last", "a_first", "m_middle"]


def test_yaml_block_no_trailing_newline():
    agent = _make_agent()
    result = agent._yaml_block({"key": "value"})
    assert not result.endswith("\n")


def test_yaml_block_handles_nested_structures():
    agent = _make_agent()
    data = {
        "steps": [
            {"step": 1, "description": "do something"},
            {"step": 2, "description": "do more"},
        ],
        "metadata": {"risk": "low"},
    }
    result = agent._yaml_block(data)
    parsed = yaml.safe_load(result)
    assert parsed == data


def test_yaml_block_handles_unicode():
    agent = _make_agent()
    data = {"message": "hello \u2014 world \u2603"}
    result = agent._yaml_block(data)
    parsed = yaml.safe_load(result)
    assert parsed == data


def test_yaml_block_handles_empty_dict():
    agent = _make_agent()
    result = agent._yaml_block({})
    parsed = yaml.safe_load(result)
    assert parsed == {}


def test_yaml_block_handles_list_input():
    agent = _make_agent()
    data = [{"a": 1}, {"b": 2}]
    result = agent._yaml_block(data)
    parsed = yaml.safe_load(result)
    assert parsed == data


# ---------------------------------------------------------------------------
# Round-trip: YAML parsing (simulates planner/testgen parse_response)
# ---------------------------------------------------------------------------

def test_yaml_roundtrip_plan():
    """Verify a YAML plan can be emitted and parsed back identically."""
    plan = {
        "steps": [
            {"step_number": 1, "description": "modify file", "files": ["a.py"], "action": "modify"},
        ],
        "files_likely_affected": ["a.py"],
        "requires_core_change": False,
        "risk_level": "low",
        "risk_notes": "safe change",
        "test_strategy": ["run pytest"],
        "estimated_complexity": "small",
        "dependencies_affected": False,
        "public_api_changed": False,
        "self_review_notes": "looks good",
    }
    yaml_str = yaml.dump(plan, default_flow_style=False, sort_keys=False)
    parsed = yaml.safe_load(yaml_str)
    assert parsed == plan


def test_yaml_roundtrip_testgen():
    """Verify a YAML test generation response round-trips."""
    response = {
        "test_file": "tests/test_foo.py",
        "content": "def test_foo():\n    assert True\n",
        "description": "basic test",
    }
    yaml_str = yaml.dump(response, default_flow_style=False, sort_keys=False)
    parsed = yaml.safe_load(yaml_str)
    assert parsed == response
