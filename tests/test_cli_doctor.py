from unittest.mock import patch

from typer.testing import CliRunner

from glitchlab.cli import app


runner = CliRunner()


@patch("glitchlab.registry.AGENT_REGISTRY", {"demo-agent": object()})
@patch("glitchlab.step_handlers.STEP_HANDLERS", {"demo-step": object()})
def test_doctor_reports_patched_registry_and_step_handlers():
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "demo-agent" in result.output
    assert "demo-step" in result.output


@patch("glitchlab.registry.AGENT_REGISTRY", {})
@patch("glitchlab.step_handlers.STEP_HANDLERS", {})
def test_doctor_handles_empty_registry_and_step_handlers():
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
