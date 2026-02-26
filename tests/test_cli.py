from typer.testing import CliRunner
from glitchlab.cli import app
from glitchlab import __version__

runner = CliRunner()

def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"GLITCHLAB v{__version__}" in result.stdout
