"""Tests for TaskState from glitchlab/controller.py."""

import json

from glitchlab.controller import TaskState, StepState


def _make_task_state(**overrides) -> TaskState:
    """Create a TaskState with sensible defaults and optional overrides."""
    defaults = {
        "task_id": "task-001",
        "objective": "Fix the bug",
        "mode": "evolution",
        "risk_level": "medium",
        "plan_steps": [
            StepState(step_number=1, description="Edit file", files=["main.py"], action="modify"),
        ],
        "files_in_scope": ["main.py", "utils.py"],
        "estimated_complexity": "medium",
        "files_modified": ["main.py"],
        "files_created": ["new_test.py"],
        "implementation_summary": "Fixed the null check",
        "last_error": "AssertionError on line 42",
        "debug_attempts": 2,
        "previous_fixes": [
            {"attempt": 1, "fix": "try A"},
            {"attempt": 2, "fix": "try B"},
            {"attempt": 3, "fix": "try C"},
        ],
        "security_verdict": "pass",
        "version_bump": "patch",
    }
    defaults.update(overrides)
    return TaskState(**defaults)


def test_summary_planner():
    """to_agent_summary('planner') returns task_id, objective, mode, risk_level, previous_fixes."""
    state = _make_task_state()
    summary = state.to_agent_summary("planner")

    assert summary["task_id"] == "task-001"
    assert summary["objective"] == "Fix the bug"
    assert summary["mode"] == "evolution"
    assert summary["risk_level"] == "medium"
    assert "previous_fixes" in summary
    # Planner gets last 3 fixes
    assert len(summary["previous_fixes"]) == 3
    # Should NOT include implementer/debugger fields
    assert "files_modified" not in summary
    assert "plan_steps" not in summary


def test_summary_implementer():
    """to_agent_summary('implementer') includes plan_steps, files_in_scope, estimated_complexity."""
    state = _make_task_state()
    summary = state.to_agent_summary("implementer")

    assert "plan_steps" in summary
    assert len(summary["plan_steps"]) == 1
    assert summary["plan_steps"][0]["description"] == "Edit file"
    assert summary["files_in_scope"] == ["main.py", "utils.py"]
    assert summary["estimated_complexity"] == "medium"
    # Base fields present
    assert summary["task_id"] == "task-001"


def test_summary_debugger():
    """to_agent_summary('debugger') includes debug-specific fields, previous_fixes capped at 2."""
    state = _make_task_state()
    summary = state.to_agent_summary("debugger")

    assert summary["files_modified"] == ["main.py"]
    assert summary["files_created"] == ["new_test.py"]
    assert summary["last_error"] == "AssertionError on line 42"
    assert summary["debug_attempts"] == 2
    # Debugger gets last 2 fixes only
    assert len(summary["previous_fixes"]) == 2
    assert summary["previous_fixes"][0]["attempt"] == 2
    assert summary["previous_fixes"][1]["attempt"] == 3


def test_summary_security():
    """to_agent_summary('security') includes files_modified, files_created, implementation_summary."""
    state = _make_task_state()
    summary = state.to_agent_summary("security")

    assert summary["files_modified"] == ["main.py"]
    assert summary["files_created"] == ["new_test.py"]
    assert summary["implementation_summary"] == "Fixed the null check"


def test_summary_release():
    """to_agent_summary('release') includes files_modified, implementation_summary, security_verdict."""
    state = _make_task_state()
    summary = state.to_agent_summary("release")

    assert summary["files_modified"] == ["main.py"]
    assert summary["implementation_summary"] == "Fixed the null check"
    assert summary["security_verdict"] == "pass"
    # Should NOT include files_created
    assert "files_created" not in summary


def test_summary_archivist():
    """to_agent_summary('archivist') includes plan_steps, files_modified, implementation_summary, version_bump."""
    state = _make_task_state()
    summary = state.to_agent_summary("archivist")

    assert "plan_steps" in summary
    assert summary["files_modified"] == ["main.py"]
    assert summary["implementation_summary"] == "Fixed the null check"
    assert summary["version_bump"] == "patch"


def test_summary_testgen():
    """to_agent_summary('testgen') includes files_modified, files_created, implementation_summary."""
    state = _make_task_state()
    summary = state.to_agent_summary("testgen")

    assert summary["files_modified"] == ["main.py"]
    assert summary["files_created"] == ["new_test.py"]
    assert summary["implementation_summary"] == "Fixed the null check"


def test_summary_unknown_role_returns_base_only():
    """to_agent_summary with unknown role returns base fields only."""
    state = _make_task_state()
    summary = state.to_agent_summary("unknown_agent")

    assert summary == {
        "task_id": "task-001",
        "objective": "Fix the bug",
        "mode": "evolution",
        "risk_level": "medium",
    }


def test_mark_phase_no_duplicates():
    """mark_phase adds to completed_phases without duplicates."""
    state = _make_task_state()

    state.mark_phase("plan")
    state.mark_phase("implement")
    state.mark_phase("plan")  # duplicate

    assert state.completed_phases == ["plan", "implement"]


def test_persist_writes_json(tmp_path):
    """persist writes task_state.json to the workspace .glitchlab directory."""
    state = _make_task_state()
    state.persist(tmp_path)

    state_file = tmp_path / ".glitchlab" / "task_state.json"
    assert state_file.exists()

    data = json.loads(state_file.read_text())
    assert data["task_id"] == "task-001"
    assert data["objective"] == "Fix the bug"
    assert data["mode"] == "evolution"
