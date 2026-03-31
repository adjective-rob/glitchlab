from typer.testing import CliRunner

from glitchlab.cli import app
from glitchlab.identity import __codename__, __version__


def test_version_flag_prints_version_and_exits_before_command_execution():
    runner = CliRunner()

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert f"{__codename__} v{__version__}" in result.stdout
    assert "Usage:" not in result.stdout
