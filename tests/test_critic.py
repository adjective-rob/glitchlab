"""Tests for the CriticAgent module."""

from unittest.mock import MagicMock

from glitchlab.agents import AgentContext
from glitchlab.agents.critic import CriticAgent, CriticResponse
from glitchlab.router import RouterResponse


def _make_context(**overrides):
    defaults = dict(
        task_id="TEST-1",
        objective="Add a greeting function",
        repo_path="/tmp/repo",
        working_dir="/tmp/repo",
        previous_output={
            "plan_steps": [{"step_number": 1, "description": "Add greet()"}],
            "files_modified": ["src/greet.py"],
            "files_created": [],
            "implementation_summary": "Added greet() function.",
        },
        file_context={"src/greet.py": "def greet(name):\n    return f'Hello {name}'\n"},
    )
    defaults.update(overrides)
    return AgentContext(**defaults)


def test_critic_agent_has_correct_role():
    router = MagicMock()
    agent = CriticAgent(router)
    assert agent.role == "critic"


def test_critic_parse_response_pass():
    router = MagicMock()
    agent = CriticAgent(router)
    ctx = _make_context()

    response = RouterResponse(
        content='{"verdict": "pass", "issues": [], "summary": "Looks good."}',
        model="gemini/gemini-3-flash-preview",
        tokens_used=100,
        cost=0.001,
    )

    result = agent.parse_response(response, ctx)
    assert result["verdict"] == "pass"
    assert result["issues"] == []
    assert result["_agent"] == "critic"


def test_critic_parse_response_with_issues():
    router = MagicMock()
    agent = CriticAgent(router)
    ctx = _make_context()

    response = RouterResponse(
        content="""{
            "verdict": "revise",
            "issues": [
                {"severity": "high", "file": "src/greet.py", "description": "Missing None check", "suggestion": "Add if name is None guard"}
            ],
            "summary": "One high-severity issue."
        }""",
        model="gemini/gemini-3-flash-preview",
        tokens_used=150,
        cost=0.002,
    )

    result = agent.parse_response(response, ctx)
    assert result["verdict"] == "revise"
    assert len(result["issues"]) == 1
    assert result["issues"][0]["severity"] == "high"


def test_critic_parse_response_invalid_json():
    router = MagicMock()
    agent = CriticAgent(router)
    ctx = _make_context()

    response = RouterResponse(
        content="not valid json at all",
        model="gemini/gemini-3-flash-preview",
        tokens_used=50,
        cost=0.0005,
    )

    result = agent.parse_response(response, ctx)
    assert result["verdict"] == "pass"  # Safe fallback
    assert result["parse_error"] is True


def test_critic_parse_response_strips_code_fences():
    router = MagicMock()
    agent = CriticAgent(router)
    ctx = _make_context()

    response = RouterResponse(
        content='```json\n{"verdict": "warn", "issues": [], "summary": "Minor nits."}\n```',
        model="gemini/gemini-3-flash-preview",
        tokens_used=80,
        cost=0.001,
    )

    result = agent.parse_response(response, ctx)
    assert result["verdict"] == "warn"
    assert result["summary"] == "Minor nits."


def test_critic_build_messages_includes_file_context():
    router = MagicMock()
    agent = CriticAgent(router)
    ctx = _make_context()

    messages = agent.build_messages(ctx)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "src/greet.py" in messages[1]["content"]
    assert "def greet" in messages[1]["content"]


def test_critic_response_model_validates():
    valid = CriticResponse(verdict="pass", issues=[], summary="All good.")
    assert valid.verdict == "pass"

    with_issues = CriticResponse(
        verdict="revise",
        issues=[{"severity": "critical", "file": "x.py", "description": "bug"}],
        summary="Found a bug.",
    )
    assert len(with_issues.issues) == 1
