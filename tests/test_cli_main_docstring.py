from glitchlab.cli import main


def test_main_docstring_describes_global_cli_callback():
    assert main.__doc__ == "Typer app callback for global CLI options such as ``--version``."