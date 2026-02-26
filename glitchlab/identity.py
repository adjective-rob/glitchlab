from rich.console import Console
from glitchlab import __codename__, __tagline__, __version__

BANNER = r"""
   ▄████  ██▓     ██▓▄▄▄█████▓ ▄████▄   ██░ ██  ██▓    ▄▄▄       ▄▄▄▄
  ██▒ ▀█▒▓██▒    ▓██▒▓  ██▒ ▓▒▒██▀ ▀█  ▓██░ ██▒▓██▒   ▒████▄    ▓█████▄
 ▒██░▄▄▄░▒██░    ▒██▒▒ ▓██░ ▒░▒▓█    ▄ ▒██▀▀██░▒██░   ▒██  ▀█▄  ▒██▒ ▄██
 ░▓█  ██▓▒██░    ░██░░ ▓██▓ ░ ▒▓▓▄ ▄██▒░▓█ ░██ ▒██░   ░██▄▄▄▄██ ▒██░█▀
 ░▒▓███▀▒░██████▒░██░  ▒██▒ ░ ▒ ▓███▀ ░░▓█▒░██▓░██████▒▓█   ▓██▒░▓█  ▀█▓
  ░▒   ▒ ░ ▒░▓  ░░▓    ▒ ░░   ░ ░▒ ▒  ░ ▒ ░░▒░▒░ ▒░▓  ░▒▒   ▓▒█░░▒▓███▀▒
"""

def print_banner(console: Console) -> None:
    console.print(f"[bright_green]{BANNER}[/]")
    console.print(f"  [dim]v{__version__} — {__tagline__}[/]\n")
