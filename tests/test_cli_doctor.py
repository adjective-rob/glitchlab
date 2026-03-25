import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from glitchlab.cli import app

runner = CliRunner()

@patch('glitchlab.cli.load_config')
@patch('glitchlab.cli.AGENT_REGISTRY', {'planner': MagicMock(), 'implementer': MagicMock()})
@patch('glitchlab.cli.STEP_HANDLERS', {'planner': MagicMock()})
def test_doctor_command_fails_when_missing_handler(mock_load_config):
    # Setup mock config with a pipeline step that is missing a handler
    mock_config = MagicMock()
    step1 = MagicMock()
    step1.agent_role = 'planner'
    step2 = MagicMock()
    step2.agent_role = 'implementer'
    mock_config.pipeline = [step1, step2]
    mock_load_config.return_value = mock_config

    result = runner.invoke(app, ["doctor"])
    
    assert result.exit_code == 1
    assert "planner" in result.stdout
    assert "implementer" in result.stdout

@patch('glitchlab.cli.load_config')
@patch('glitchlab.cli.AGENT_REGISTRY', {'planner': MagicMock(), 'implementer': MagicMock()})
@patch('glitchlab.cli.STEP_HANDLERS', {'planner': MagicMock(), 'implementer': MagicMock()})
def test_doctor_command_succeeds_when_all_present(mock_load_config):
    # Setup mock config with a pipeline step that has all handlers
    mock_config = MagicMock()
    step1 = MagicMock()
    step1.agent_role = 'planner'
    step2 = MagicMock()
    step2.agent_role = 'implementer'
    mock_config.pipeline = [step1, step2]
    mock_load_config.return_value = mock_config

    result = runner.invoke(app, ["doctor"])
    
    assert result.exit_code == 0
    assert "planner" in result.stdout
    assert "implementer" in result.stdout
