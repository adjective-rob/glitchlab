from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
from typer.testing import CliRunner
from glitchlab.cli import app

runner = CliRunner()


def test_glitchlab_entrypoint_accepts_run_invocation(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    with tempfile.TemporaryDirectory() as home_dir:
        monkeypatch.setenv("HOME", home_dir)
        monkeypatch.setenv("GLITCHLAB_OFFLINE", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        result = runner.invoke(app, ["run", "demo-task", "--repo", str(repo_dir)])

    assert result.exit_code != 2
    assert "Usage:" not in result.output

FAKE_ENTRIES = [
    {
        "task_id": "TASK_001",
        "status": "pr_created",
        "budget": {
            "estimated_cost": 0.12,
            "total_tokens": 5000,
            "role_usage": {"planner": 1000, "implementer": 3000, "tester": 1000},
        },
        "quality_score": {"score": 85},
        "events_summary": {"fix_attempts": 1, "plan_steps": 4},
    },
    {
        "task_id": "TASK_002",
        "status": "committed",
        "budget": {
            "estimated_cost": 0.25,
            "total_tokens": 9000,
            "role_usage": {"planner": 1500, "implementer": 6000, "tester": 1500},
        },
        "quality_score": {"score": 72},
        "events_summary": {"fix_attempts": 3, "plan_steps": 5},
    },
]


@patch("glitchlab.cli.TaskHistory")
def test_compare_basic(mock_history_cls):
    mock_hist = MagicMock()
    mock_hist.get_all.return_value = FAKE_ENTRIES
    mock_history_cls.return_value = mock_hist

    result = runner.invoke(app, ["compare", "--repo", ".", "TASK_001", "TASK_002"])
    assert result.exit_code == 0
    assert "TASK_001" in result.output
    assert "TASK_002" in result.output
    assert "Comparing" in result.output
    assert "Planner" in result.output
    assert "Divergence summary" in result.output


@patch("glitchlab.cli.TaskHistory")
def test_compare_task_not_found(mock_history_cls):
    mock_hist = MagicMock()
    mock_hist.get_all.return_value = FAKE_ENTRIES
    mock_history_cls.return_value = mock_hist

    result = runner.invoke(app, ["compare", "--repo", ".", "TASK_001", "TASK_999"])
    assert result.exit_code == 1
    assert "not found" in result.output


@patch("glitchlab.cli.TaskHistory")
def test_compare_both_not_found(mock_history_cls):
    mock_hist = MagicMock()
    mock_hist.get_all.return_value = []
    mock_history_cls.return_value = mock_hist

    result = runner.invoke(app, ["compare", "--repo", ".", "X", "Y"])
    assert result.exit_code == 1
    assert "not found" in result.output
