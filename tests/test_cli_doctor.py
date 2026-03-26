from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from glitchlab.cli import app

runner = CliRunner()


@patch("glitchlab.config_loader.load_config")
@patch("glitchlab.registry.AGENT_REGISTRY", {"planner": MagicMock(), "implementer": MagicMock()})
@patch("glitchlab.step_handlers.STEP_HANDLERS", {"planner": MagicMock(), "implementer": MagicMock()})
def test_doctor_command_succeeds_when_all_present(mock_load_config):
    mock_config = MagicMock()
    step1 = MagicMock(); step1.agent_role = "planner"
    step2 = MagicMock(); step2.agent_role = "implementer"
    mock_config.pipeline = [step1, step2]
    mock_load_config.return_value = mock_config

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "planner" in result.output
    assert "implementer" in result.output


@patch("glitchlab.config_loader.load_config")
@patch("glitchlab.registry.AGENT_REGISTRY", {"planner": MagicMock(), "implementer": MagicMock()})
@patch("glitchlab.step_handlers.STEP_HANDLERS", {"planner": MagicMock()})
def test_doctor_command_fails_when_missing_handler(mock_load_config):
    mock_config = MagicMock()
    step1 = MagicMock(); step1.agent_role = "planner"
    step2 = MagicMock(); step2.agent_role = "implementer"
    mock_config.pipeline = [step1, step2]
    mock_load_config.return_value = mock_config

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "planner" in result.output
    assert "implementer" in result.output