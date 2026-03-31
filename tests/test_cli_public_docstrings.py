from glitchlab import cli


def test_public_cli_functions_have_descriptive_docstrings():
    assert cli.version_callback.__doc__ == "Print the CLI version for the global --version flag and exit immediately."
    assert cli.main.__doc__ == "Launch the Typer application as the CLI entry point."
    assert cli.run.__doc__ == "Run the main CLI workflow that resolves a task and executes the agent pipeline."
