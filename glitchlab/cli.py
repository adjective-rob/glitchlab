"""
GLITCHLAB CLI — The Interface

Three modes:
  1. glitchlab run --repo <path> --issue <num>     (GitHub issue)
  2. glitchlab run --repo <path> --local-task       (YAML file)
  3. glitchlab interactive --repo <path>            (Human-in-the-loop)

Plus utilities:
  - glitchlab status        (check config + API keys)
  - glitchlab init <path>   (bootstrap .glitchlab in a repo)
  - glitchlab batch         (parallel task execution)
  - glitchlab history       (view previous runs)
  - glitchlab audit         (scan for new tasks)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from glitchlab.identity import __codename__, __tagline__, __version__, BANNER
from glitchlab.config_loader import load_config, validate_api_keys
from glitchlab.controller import Controller, Task
from glitchlab.history import TaskHistory
from glitchlab.parallel import run_parallel
from glitchlab.prelude import PreludeContext

# Load .env from current directory or home
load_dotenv()
load_dotenv(Path.home() / ".glitchlab" / ".env")

app = typer.Typer(
    name="glitchlab",
    help=f"{__codename__} — {__tagline__}\nThe Agentic Dev Engine.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"{__codename__} v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
):
    pass


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def _print_banner():
    console.print(f"[bright_green]{BANNER}[/]")
    console.print(f"  [dim]v{__version__} — {__tagline__}[/]\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    repo: Path = typer.Option(..., "--repo", "-r", help="Path to the target repository"),
    issue: Optional[int] = typer.Option(None, "--issue", "-i", help="GitHub issue number"),
    local_task: bool = typer.Option(False, "--local-task", "-l", help="Use local task YAML"),
    task_file: Optional[Path] = typer.Option(None, "--task-file", "-f", help="Path to task YAML"),
    allow_core: bool = typer.Option(False, "--allow-core", help="Allow modifications to protected core paths"),
    auto_approve: bool = typer.Option(False, "--auto-approve", "-y", help="Skip human intervention gates"),
    test_cmd: Optional[str] = typer.Option(None, "--test", "-t", help="Test command to run (e.g. 'cargo test')"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Run GLITCHLAB on a task."""
    _print_banner()
    _configure_logging(verbose)

    repo = repo.resolve()
    if not repo.exists():
        console.print(f"[red]Repository not found: {repo}[/]")
        raise typer.Exit(1)

    config = load_config(repo)

    # Resolve task
    if issue:
        console.print(f"[cyan]Fetching GitHub issue #{issue}...[/]")
        task = Task.from_github_issue(repo, issue)
    elif local_task or task_file:
        # Check queue first, then root tasks dir
        tf = task_file or (repo / ".glitchlab" / "tasks" / "queue" / "next.yaml")
        if not tf.exists():
            tf = (repo / ".glitchlab" / "tasks" / "next.yaml")
            
        if not tf.exists():
            console.print(f"[red]Task file not found: {tf}[/]")
            raise typer.Exit(1)
        task = Task.from_yaml(tf)
    else:
        console.print("[red]Specify --issue or --local-task[/]")
        raise typer.Exit(1)

    # Auto-detect test command if not provided
    if not test_cmd:
        test_cmd = _detect_test_command(repo)

    controller = Controller(
        repo_path=repo,
        config=config,
        allow_core=allow_core,
        auto_approve=auto_approve,
        test_command=test_cmd,
    )

    result = controller.run(task)

    # Final status
    status = result.get("status", "unknown")
    status_color = {
        "pr_created": "green",
        "committed": "yellow",
        "interrupted": "yellow",
    }.get(status, "red")

    console.print(f"\n[bold {status_color}]Status: {status}[/]")


@app.command()
def interactive(
    repo: Path = typer.Option(..., "--repo", "-r", help="Path to the target repository"),
    allow_core: bool = typer.Option(False, "--allow-core"),
    test_cmd: Optional[str] = typer.Option(None, "--test", "-t"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Interactive mode — describe what you want, review the plan, approve execution."""
    _print_banner()
    _configure_logging(verbose)

    repo = repo.resolve()
    config = load_config(repo)

    console.print("[bold]Describe what you want GLITCHLAB to do:[/]")
    objective = typer.prompt(">>")

    if not objective.strip():
        console.print("[red]No objective provided.[/]")
        raise typer.Exit(1)

    task = Task.from_interactive(objective)

    if not test_cmd:
        test_cmd = _detect_test_command(repo)

    controller = Controller(
        repo_path=repo,
        config=config,
        allow_core=allow_core,
        test_command=test_cmd,
    )

    result = controller.run(task)
    status = result.get("status", "unknown")
    console.print(f"\n[bold]Status: {status}[/]")


@app.command()
def status(
    repo: Optional[Path] = typer.Option(None, "--repo", "-r"),
):
    """Check GLITCHLAB configuration and readiness."""
    _print_banner()

    # API Keys
    keys = validate_api_keys()
    key_table = Table(title="API Keys", border_style="cyan")
    key_table.add_column("Key")
    key_table.add_column("Status")

    for key, available in keys.items():
        status_str = "[green]✓ Available[/]" if available else "[red]✗ Missing[/]"
        key_table.add_row(key, status_str)

    console.print(key_table)

    # Config
    if repo:
        config = load_config(repo.resolve())
        console.print(f"\n[bold]Routing:[/]")
        console.print(f"  Planner:     {config.routing.planner}")
        console.print(f"  Implementer: {config.routing.implementer}")
        console.print(f"  Debugger:    {config.routing.debugger}")
        console.print(f"  Security:    {config.routing.security}")
        console.print(f"  Release:     {config.routing.release}")

        console.print(f"\n[bold]Limits:[/]")
        console.print(f"  Max fix attempts: {config.limits.max_fix_attempts}")
        console.print(f"  Max tokens/task:  {config.limits.max_tokens_per_task:,}")
        console.print(f"  Max $/task:       ${config.limits.max_dollars_per_task}")

        if config.boundaries.protected_paths:
            console.print(f"\n[bold]Protected paths:[/]")
            for p in config.boundaries.protected_paths:
                console.print(f"  🔒 {p}")

    # Tools
    tools_table = Table(title="System Tools", border_style="cyan")
    tools_table.add_column("Tool")
    tools_table.add_column("Status")

    import shutil
    for tool in ["git", "gh", "cargo", "python3", "node", "prelude"]:
        found = shutil.which(tool)
        s = f"[green]✓ {found}[/]" if found else "[dim]✗ Not found[/]"
        tools_table.add_row(tool, s)

    console.print(tools_table)

    # Prelude context
    if repo:
        prelude = PreludeContext(repo.resolve())
        prelude_table = Table(title="Prelude Context", border_style="magenta")
        prelude_table.add_column("Property")
        prelude_table.add_column("Value")

        prelude_table.add_row("CLI installed", "✓" if prelude.cli_available else "✗")
        prelude_table.add_row(".context/ exists", "✓" if prelude.context_exists else "✗")

        if prelude.context_exists:
            summary = prelude.summary()
            prelude_table.add_row("Context files", ", ".join(summary.get("files", [])))
            prelude_table.add_row("ADRs", str(summary.get("decisions_count", 0)))
            if "project_name" in summary:
                prelude_table.add_row("Project", summary["project_name"])
            if "language" in summary:
                prelude_table.add_row("Language", summary["language"])

        console.print(prelude_table)

        if not prelude.available:
            console.print(
                "[dim]Install Prelude for richer agent context: "
                "npm install -g prelude-context[/]"
            )


@app.command()
def init(
    repo: Optional[Path] = typer.Argument(None, help="Path to repository"),
):
    """Initialize .glitchlab directory in a repository."""
    _print_banner()

    repo = (repo or Path.cwd()).resolve()
    gl_dir = repo / ".glitchlab"
    gl_dir.mkdir(exist_ok=True)
    (gl_dir / "tasks").mkdir(exist_ok=True)
    (gl_dir / "tasks" / "queue").mkdir(exist_ok=True) # Permanent Fix: Ensure queue exists
    (gl_dir / "logs").mkdir(exist_ok=True)
    (gl_dir / "worktrees").mkdir(exist_ok=True)

    # Create default config
    config_path = gl_dir / "config.yaml"
    if not config_path.exists():
        config_path.write_text("""# GLITCHLAB repo-level config overrides
# These merge with the built-in defaults.

# Override routing for this specific project:
# routing:
#   implementer: "anthropic/claude-sonnet-4-20250514"

# Set project-specific boundaries:
# boundaries:
#   protected_paths:
#     - "crates/zephyr-core"
#     - "crates/zephyr-envelope"

# Adjust limits:
# limits:
#   max_fix_attempts: 3
#   max_dollars_per_task: 5.0
""")

    # Create example task
    task_path = gl_dir / "tasks" / "example.yaml"
    if not task_path.exists():
        task_path.write_text("""id: example-001
objective: "Add a --verbose flag to the CLI"
constraints:
  - "No new dependencies"
  - "Must not change existing flag behavior"
acceptance:
  - "Tests pass"
  - "New test covers the flag"
  - "Help text updated"
risk: low
""")

    # Add to .gitignore
    gitignore = repo / ".gitignore"
    ignore_entries = [".glitchlab/worktrees/", ".glitchlab/logs/"]
    if gitignore.exists():
        content = gitignore.read_text()
        additions = [e for e in ignore_entries if e not in content]
        if additions:
            with open(gitignore, "a") as f:
                f.write("\n# GLITCHLAB\n")
                for e in additions:
                    f.write(f"{e}\n")
    else:
        gitignore.write_text("# GLITCHLAB\n" + "\n".join(ignore_entries) + "\n")

    console.print(f"[green]✅ Initialized GLITCHLAB in {gl_dir}[/]")
    console.print(f"  Config:  {config_path}")
    console.print(f"  Tasks:   {gl_dir / 'tasks'}")
    console.print(f"  Example: {task_path}")

    # Bootstrap Prelude if available
    prelude = PreludeContext(repo)
    if prelude.cli_available:
        if not prelude.context_exists:
            console.print("\n[magenta]📋 Prelude detected — initializing project context...[/]")
            if prelude.init():
                console.print("[green]  ✅ .context/ created — your agents now see the full project[/]")
                console.print("  [dim]Add decisions: prelude decision[/]")
                console.print("  [dim]Update context: prelude update[/]")
            else:
                console.print("[yellow]  ⚠ Prelude init failed — agents will work without project context[/]")
        else:
            console.print(f"\n[dim]📋 Prelude context already exists at {prelude.context_dir}[/]")
    else:
        console.print(
            "\n[dim]💡 Install Prelude for richer agent context: "
            "npm install -g prelude-context[/]"
        )


@app.command()
def batch(
    repo: Path = typer.Option(..., "--repo", "-r", help="Path to the target repository"),
    tasks_dir: Optional[Path] = typer.Option(None, "--tasks-dir", "-d", help="Directory of task YAMLs"),
    workers: int = typer.Option(3, "--workers", "-w", help="Max concurrent tasks"),
    allow_core: bool = typer.Option(False, "--allow-core"),
    test_cmd: Optional[str] = typer.Option(None, "--test", "-t"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run multiple tasks in parallel (auto-approve mode)."""
    _print_banner()
    _configure_logging(verbose)

    repo = repo.resolve()
    # Permanent Fix: Default to the 'queue' subfolder used by the Auditor
    td = tasks_dir or (repo / ".glitchlab" / "tasks" / "queue")

    if not td.exists():
        console.print(f"[red]Tasks directory not found: {td}[/]")
        raise typer.Exit(1)

    task_files = sorted(td.glob("*.yaml")) + sorted(td.glob("*.yml"))
    # Exclude example files
    task_files = [f for f in task_files if "example" not in f.name.lower()]

    if not task_files:
        console.print(f"[red]No task files found in {td}[/]")
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(task_files)} tasks in {td}[/]")
    for tf in task_files:
        console.print(f"  [dim]{tf.name}[/]")

    if not test_cmd:
        test_cmd = _detect_test_command(repo)

    results = run_parallel(
        repo_path=repo,
        task_files=task_files,
        max_workers=workers,
        allow_core=allow_core,
        test_command=test_cmd,
    )

    # Exit code based on results
    failures = sum(1 for r in results if r.get("status") not in ("pr_created", "committed"))
    if failures:
        raise typer.Exit(1)


@app.command()
def history(
    repo: Path = typer.Option(..., "--repo", "-r", help="Path to the target repository"),
    count: int = typer.Option(10, "--count", "-n", help="Number of entries to show"),
    stats: bool = typer.Option(False, "--stats", "-s", help="Show aggregate statistics"),
):
    """View task history and statistics."""
    _print_banner()

    repo = repo.resolve()
    hist = TaskHistory(repo)

    if stats:
        s = hist.get_stats()
        if s["total_runs"] == 0:
            console.print("[dim]No history yet.[/]")
            return

        stats_table = Table(title="GLITCHLAB Statistics", border_style="cyan")
        stats_table.add_column("Metric")
        stats_table.add_column("Value")

        stats_table.add_row("Total runs", str(s["total_runs"]))
        stats_table.add_row("Success rate", f"{s['success_rate']}%")
        stats_table.add_row("Total cost", f"${s['total_cost']:.4f}")
        stats_table.add_row("Total tokens", f"{s['total_tokens']:,}")
        stats_table.add_row("Avg cost/run", f"${s['avg_cost_per_run']:.4f}")

        console.print(stats_table)

        if s.get("statuses"):
            status_table = Table(title="Status Breakdown", border_style="dim")
            status_table.add_column("Status")
            status_table.add_column("Count")
            for status_name, cnt in sorted(s["statuses"].items(), key=lambda x: -x[1]):
                status_table.add_row(status_name, str(cnt))
            console.print(status_table)

        return

    # Show recent entries
    entries = hist.get_recent(count)
    if not entries:
        console.print("[dim]No history yet. Run some tasks first.[/]")
        return

    table = Table(title=f"Recent Runs (last {count})", border_style="cyan")
    table.add_column("Time", style="dim")
    table.add_column("Task")
    table.add_column("Status")
    table.add_column("Cost")
    table.add_column("Notes")

    for entry in reversed(entries):
        ts = entry.get("timestamp", "?")[:19]
        status = entry.get("status", "?")
        color = {"pr_created": "green", "committed": "yellow"}.get(status, "red")
        budget = entry.get("budget", {})
        cost = f"${budget.get('estimated_cost', 0):.4f}"

        notes = ""
        events = entry.get("events_summary", {})
        if events.get("fix_attempts", 0) > 0:
            notes += f"fixes:{events['fix_attempts']} "
        if events.get("security_verdict"):
            notes += f"sec:{events['security_verdict']} "
        if entry.get("error"):
            notes += entry["error"][:40]

        table.add_row(ts, entry.get("task_id", "?"), f"[{color}]{status}[/]", cost, notes)

    console.print(table)

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

@app.command()
def audit(
    repo: Path = typer.Option(..., help="Path to the repository to audit"),
    kind: str = typer.Option(None, help="Filter by finding kind: missing_doc, todo, complex_function"),
    dry_run: bool = typer.Option(False, help="Print findings without generating task files"),
    output_dir: Path = typer.Option(None, help="Directory to write task YAMLs (default: .glitchlab/tasks/queue)"),
):
    """
    Scan a repository for actionable findings and generate GLITCHLAB task files.
    """
    from glitchlab.auditor import Scanner, TaskWriter
    from glitchlab.router import Router

    repo_path = repo.resolve()
    if not repo_path.exists():
        console.print(f"[red]Repository not found: {repo_path}[/]")
        raise typer.Exit(1)

    console.print(f"\n[bold dim]🔍 [AUDITOR] Scanning {repo_path.name}...[/]")
    scanner = Scanner(repo_path)
    result = scanner.scan()

    summary = result.summary()
    console.print(f"  [dim]Scanned {summary['files_scanned']} files, found {summary['total']} findings[/]")

    findings = result.findings
    if kind:
        findings = [f for f in findings if f.kind == kind]
        console.print(f"  [dim]Filtered to {len(findings)} findings of kind '{kind}'[/]")

    if not findings:
        console.print("[green]✅ No findings. Codebase looks clean![/]")
        return

    table = Table(title="Findings", border_style="yellow")
    table.add_column("Kind", style="dim")
    table.add_column("File")
    table.add_column("Line", style="dim")
    table.add_column("Description")
    table.add_column("Severity")

    for f in findings[:50]:
        color = {"high": "red", "medium": "yellow", "low": "dim"}.get(f.severity, "dim")
        table.add_row(f.kind, f.file, str(f.line), f.description[:80], f"[{color}]{f.severity}[/]")

    console.print(table)

    if len(findings) > 50:
        console.print(f"[dim]... and {len(findings) - 50} more[/]")

    if dry_run:
        console.print("\n[yellow]Dry run — no task files written.[/]")
        return

    out_dir = output_dir or (repo_path / ".glitchlab" / "tasks" / "queue")
    console.print(f"\n[bold dim]📝 [AUDITOR] Generating task files → {out_dir}[/]")

    config = load_config(repo_path)
    router = Router(config)
    writer = TaskWriter(router, out_dir)

    result.findings = findings
    written = writer.write_tasks(result)

    console.print(Panel(
        "\n".join(f"  {p.name}" for p in written),
        title=f"✅ {len(written)} task files written",
        border_style="green",
    ))
    console.print(f"\nRun tasks with: [bold]glitchlab batch --repo {repo_path} --tasks-dir {out_dir}[/]")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_test_command(repo: Path) -> str | None:
    """Auto-detect the test command based on repo contents."""
    if (repo / "Cargo.toml").exists():
        return "cargo test"
    if (repo / "package.json").exists():
        return "npm test"
    if (repo / "pyproject.toml").exists() or (repo / "setup.py").exists():
        return "python -m pytest"
    if (repo / "go.mod").exists():
        return "go test ./..."
    if (repo / "Makefile").exists():
        return "make test"
    return None


def _configure_logging(verbose: bool) -> None:
    logger.remove()
    if verbose:
        logger.add(
            lambda msg: console.print(f"[dim]{msg}[/]", highlight=False),
            level="DEBUG",
            format="{time:HH:mm:ss} | {level:<7} | {message}",
        )
    else:
        logger.add(
            lambda msg: console.print(f"[dim]{msg}[/]", highlight=False),
            level="WARNING",
            format="{message}",
        )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()