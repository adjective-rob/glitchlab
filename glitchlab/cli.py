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
from glitchlab.audit_logger import AuditLogger  # <--- NEW: Import the Zephyr subscriber

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
    """Typer app callback for global CLI options such as ``--version``."""
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
    surgical: bool = typer.Option(False, "--surgical", help="Run surgical pipeline"),
    auto_merge: bool = typer.Option(False, "--auto-merge", help="Automatically squash and merge the PR if successful"),
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

    if auto_merge:
        config.automation.auto_merge_pr = True

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

    # --- NEW: Initialize the Zephyr Audit Logger subscriber ---
    audit_logger = AuditLogger(log_file=repo / ".glitchlab" / "logs" / "audit.jsonl")

    controller = Controller(
        repo_path=repo,
        config=config,
        allow_core=allow_core,
        auto_approve=auto_approve,
        surgical=surgical,
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
    auto_approve: bool = typer.Option(False, "--auto-approve", "-y", help="Skip human intervention gates"),
    surgical: bool = typer.Option(False, "--surgical", help="Run surgical pipeline"),
    auto_merge: bool = typer.Option(False, "--auto-merge", help="Automatically squash and merge the PR if successful"),
    test_cmd: Optional[str] = typer.Option(None, "--test", "-t"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Interactive mode — describe what you want, review the plan, approve execution."""
    _print_banner()
    _configure_logging(verbose)

    repo = repo.resolve()
    config = load_config(repo)

    if auto_merge:
        config.automation.auto_merge_pr = True

    console.print("[bold]Describe what you want GLITCHLAB to do:[/]")
    objective = typer.prompt(">>")

    if not objective.strip():
        console.print("[red]No objective provided.[/]")
        raise typer.Exit(1)

    task = Task.from_interactive(objective)

    if not test_cmd:
        test_cmd = _detect_test_command(repo)

    # --- NEW: Initialize the Zephyr Audit Logger subscriber ---
    audit_logger = AuditLogger(log_file=repo / ".glitchlab" / "logs" / "audit.jsonl")

    controller = Controller(
        repo_path=repo,
        config=config,
        allow_core=allow_core,
        auto_approve=auto_approve,
        surgical=surgical,
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
    for tool in ["git", "gh", "cargo", "python3", "node", "prelude", "zephyr"]:
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

    # Create ROADMAP template
    roadmap_path = gl_dir / "ROADMAP.md"
    if not roadmap_path.exists():
        roadmap_path.write_text("""# Project Roadmap
#
# The GLITCHLAB auditor reads this file every scan to prioritize task generation.
# Tasks that advance "Now" items are boosted. "Deferred" areas are skipped.
# Edit this file as your priorities change — the auditor picks it up automatically.

## Now
- (what you're actively working on)

## Next
- (queued work, not urgent but directional)

## Deferred
- (areas where the auditor should NOT generate tasks right now)
""")

    # Add to .gitignore
    gitignore = repo / ".gitignore"
    ignore_entries = [".glitchlab/worktrees/", ".glitchlab/tasks/",".glitchlab/logs/"]
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
    console.print(f"  Roadmap: {roadmap_path}")

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
    auto_merge: bool = typer.Option(False, "--auto-merge", help="Automatically squash and merge the PRs if successful"),
    test_cmd: Optional[str] = typer.Option(None, "--test", "-t"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run multiple tasks in parallel (auto-approve mode)."""
    _print_banner()
    _configure_logging(verbose)

    repo = repo.resolve()
    td = tasks_dir or (repo / ".glitchlab" / "tasks" / "queue")

    if not td.exists():
        console.print(f"[red]Tasks directory not found: {td}[/]")
        raise typer.Exit(1)

    raw_task_files = sorted(td.glob("*.yaml")) + sorted(td.glob("*.yml"))
    raw_task_files = [f for f in raw_task_files if "example" not in f.name.lower()]

    if not raw_task_files:
        console.print(f"[red]No task files found in {td}[/]")
        raise typer.Exit(1)

    # --- NEW: Task Priority Queue ---
    # Parse and sort tasks by risk so High Risk runs first when main is cleanest.
    tasks_with_risk = []
    for tf in raw_task_files:
        try:
            t = Task.from_yaml(tf)
            # High = 0 (first), Medium = 1, Low = 2
            weight = {"high": 0, "medium": 1, "low": 2}.get(t.risk_level, 3)
            tasks_with_risk.append((weight, tf))
        except Exception:
            tasks_with_risk.append((3, tf)) # Push parsing errors to the end
            
    tasks_with_risk.sort(key=lambda x: x[0])
    task_files = [tf for _, tf in tasks_with_risk]

    console.print(f"[cyan]Found {len(task_files)} tasks in {td} (Sorted by Risk Priority)[/]")
    for tf in task_files:
        console.print(f"  [dim]{tf.name}[/]")

    if not test_cmd:
        test_cmd = _detect_test_command(repo)

    # --- NEW: Initialize the Zephyr Audit Logger subscriber ---
    audit_logger = AuditLogger(log_file=repo / ".glitchlab" / "logs" / "audit.jsonl")

    results = run_parallel(
        repo_path=repo,
        task_files=task_files,
        max_workers=workers,
        allow_core=allow_core,
        test_command=test_cmd,
        auto_merge=auto_merge,
    )

    # Exit code based on results
    failures = sum(1 for r in results if r.get("status") not in ("pr_created", "committed", "merged"))
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
# Compare
# ---------------------------------------------------------------------------


@app.command()
def compare(
    repo: Path = typer.Option(..., "--repo", "-r", help="Path to the target repository"),
    task_a: str = typer.Argument(..., help="First task ID to compare"),
    task_b: str = typer.Argument(..., help="Second task ID to compare"),
):
    """Compare two task runs — cost, loop steps, planner accuracy, tool divergence."""
    _configure_logging(False)
    repo = repo.resolve()
    hist = TaskHistory(repo)
    all_entries = hist.get_all()
    entry_a = next((e for e in all_entries if e.get("task_id") == task_a), None)
    entry_b = next((e for e in all_entries if e.get("task_id") == task_b), None)
    if not entry_a:
        console.print(f"[red]Task '{task_a}' not found in history.[/]")
        raise typer.Exit(1)
    if not entry_b:
        console.print(f"[red]Task '{task_b}' not found in history.[/]")
        raise typer.Exit(1)

    def _budget(e: dict) -> dict:
        return e.get("budget", {})

    def _events(e: dict) -> dict:
        return e.get("events_summary", {})

    def _quality(e: dict) -> int:
        q = e.get("quality_score", {})
        return q.get("score", 0) if isinstance(q, dict) else int(q or 0)

    def _cost(e: dict) -> float:
        return _budget(e).get("estimated_cost", 0.0)

    def _tokens(e: dict) -> int:
        return _budget(e).get("total_tokens", 0)

    def _fix_attempts(e: dict) -> int:
        return _events(e).get("fix_attempts", 0)

    def _loop_steps(e: dict) -> int:
        return _budget(e).get("role_usage", {}).get("implementer", 0)

    # --- Header ---
    console.print(Panel(
        f"[bold]Comparing:[/] [cyan]{task_a}[/] vs [cyan]{task_b}[/]",
        title="⚡ GLITCHLAB Compare",
        border_style="bright_green",
    ))

    # --- Side-by-side metrics table ---
    table = Table(border_style="cyan", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column(task_a, style="cyan")
    table.add_column(task_b, style="magenta")

    def _delta(val_a: float, val_b: float, lower_is_better: bool = True) -> str:
        if val_a == val_b:
            return "="
        better = val_a < val_b if lower_is_better else val_a > val_b
        arrow = "▲" if val_a > val_b else "▼"
        color = "green" if better else "red"
        return f"[{color}]{arrow}[/]"

    cost_a, cost_b = _cost(entry_a), _cost(entry_b)
    tok_a, tok_b = _tokens(entry_a), _tokens(entry_b)
    fix_a, fix_b = _fix_attempts(entry_a), _fix_attempts(entry_b)
    q_a, q_b = _quality(entry_a), _quality(entry_b)
    status_a = entry_a.get("status", "?")
    status_b = entry_b.get("status", "?")

    table.add_row("Status", status_a, status_b)
    table.add_row("Cost", f"${cost_a:.4f} {_delta(cost_a, cost_b)}", f"${cost_b:.4f}")
    table.add_row("Tokens", f"{tok_a:,} {_delta(tok_a, tok_b)}", f"{tok_b:,}")
    table.add_row("Debug attempts", f"{fix_a} {_delta(fix_a, fix_b)}", f"{fix_b}")
    table.add_row(
        "Quality score",
        f"{q_a} {_delta(q_a, q_b, lower_is_better=False)}",
        f"{q_b}",
    )
    console.print(table)

    # --- Planner accuracy ---
    def _planner_accuracy(e: dict) -> tuple[int, int]:
        evs = e.get("events_summary", {})
        planned = evs.get("plan_steps", 0)
        return planned, evs.get("fix_attempts", 0)

    pa_a, err_a = _planner_accuracy(entry_a)
    pa_b, err_b = _planner_accuracy(entry_b)
    console.print("\n[bold]Planner:[/]")
    console.print(f"  {task_a}: {pa_a} plan steps, {err_a} fix attempt(s)")
    console.print(f"  {task_b}: {pa_b} plan steps, {err_b} fix attempt(s)")

    # --- Budget role breakdown ---
    roles_a = _budget(entry_a).get("role_usage", {})
    roles_b = _budget(entry_b).get("role_usage", {})
    all_roles = sorted(set(list(roles_a.keys()) + list(roles_b.keys())))
    if roles_a or roles_b:
        role_table = Table(title="Token usage by role", border_style="dim")
        role_table.add_column("Role")
        role_table.add_column(task_a, style="cyan")
        role_table.add_column(task_b, style="magenta")
        for role in all_roles:
            ra = roles_a.get(role, 0)
            rb = roles_b.get(role, 0)
            role_table.add_row(role, f"{ra:,}", f"{rb:,}")
        console.print(role_table)

    # --- Divergence hint ---
    console.print("\n[bold]Divergence summary:[/]")
    if cost_a > cost_b * 2:
        console.print(
            f"  [yellow]⚠  {task_a} cost {cost_a / cost_b:.1f}× more than {task_b}[/]"
        )
    if fix_a > fix_b:
        console.print(
            f"  [yellow]⚠  {task_a} needed {fix_a - fix_b} more debug attempt(s)[/]"
        )
    if fix_a == 0 and fix_b == 0 and cost_a > cost_b:
        console.print(
            "  [dim]Both converged without debug loops."
            " Cost difference likely in implementer exploration.[/]"
        )
    if q_a < 50:
        console.print(
            f"  [red]✗  {task_a} quality score is low ({q_a})"
            " — check for breaker trips or parse errors.[/]"
        )
    if q_b < 50:
        console.print(
            f"  [red]✗  {task_b} quality score is low ({q_b})"
            " — check for breaker trips or parse errors.[/]"
        )
    if q_a >= 80 and q_b >= 80:
        console.print("  [green]✓  Both runs converged cleanly.[/]")


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

@app.command()
def audit(
    repo: Path = typer.Option(..., help="Path to the repository to audit"),
    kind: str = typer.Option(None, help="Filter by finding kind (e.g. missing_doc, todo, dead_code, dependency_vuln)"),
    category: str = typer.Option(None, help="Filter by category: security, bug, test, refactor, cleanup, docs, feature"),
    scout: bool = typer.Option(False, "--scout", "-s", help="Enable Scout Brain — LLM-powered creative analysis for feature ideas"),
    dry_run: bool = typer.Option(False, help="Print findings without generating task files"),
    output_dir: Path = typer.Option(None, help="Directory to write task YAMLs (default: .glitchlab/tasks/queue)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """
    Scout — Autonomous codebase analysis engine.

    Scans your repository across multiple layers and generates
    prioritized, batch-ready GLITCHLAB task files.
    """
    from glitchlab.auditor import Scanner, TaskWriter
    from glitchlab.router import Router

    _configure_logging(verbose)

    repo_path = repo.resolve()
    if not repo_path.exists():
        console.print(f"[red]Repository not found: {repo_path}[/]")
        raise typer.Exit(1)

    layer_desc = "static + scout" if scout else "static"
    console.print(f"\n[bold cyan]SCOUT[/] [dim]Scanning {repo_path.name} [layers: {layer_desc}]...[/]")

    scanner = Scanner(repo_path)
    result = scanner.scan()

    summary = result.summary()

    # Summary panel
    summary_lines = [
        f"Files scanned: {summary['files_scanned']}",
        f"Total findings: {summary['total']}",
    ]

    console.print(Panel(
        "\n".join(summary_lines),
        title="[bold cyan]Scout Scan Results[/]",
        border_style="cyan",
    ))

    # Apply filters
    findings = result.findings
    if kind:
        findings = [f for f in findings if getattr(f, "kind", None) == kind]
        console.print(f"  [dim]Filtered to {len(findings)} findings of kind '{kind}'[/]")
    if category:
        findings = [f for f in findings if getattr(f, "category", None) == category]
        console.print(f"  [dim]Filtered to {len(findings)} findings in category '{category}'[/]")

    if not findings and not scout:
        console.print("[green]No findings. Codebase looks clean![/]")
        return

    # Findings table
    if findings:
        table = Table(title="Findings", border_style="yellow")
        table.add_column("Cat", style="dim", width=8)
        table.add_column("Kind", style="dim", width=18)
        table.add_column("File", width=35)
        table.add_column("Line", style="dim", width=5)
        table.add_column("Description")
        table.add_column("Sev", width=6)

        # Sort for display: severity desc
        display_findings = sorted(
            findings,
            key=lambda f: {"high": 0, "medium": 1, "low": 2}.get(getattr(f, "severity", "low"), 3)
        )

        for f in display_findings[:80]:
            sev_color = {"high": "red", "medium": "yellow", "low": "dim"}.get(getattr(f, "severity", "low"), "dim")
            cat_color = {"security": "red", "bug": "yellow", "feature": "green", "test": "cyan"}.get(getattr(f, "category", "refactor"), "dim")
            table.add_row(
                f"[{cat_color}]{getattr(f, 'category', 'refactor')}[/]",
                getattr(f, "kind", "unknown"),
                getattr(f, "file", "unknown"),
                str(getattr(f, "line", "?")),
                getattr(f, "description", "")[:80],
                f"[{sev_color}]{getattr(f, 'severity', 'low')}[/]",
            )

        console.print(table)

        if len(findings) > 80:
            console.print(f"[dim]... and {len(findings) - 80} more findings[/]")

    if dry_run and not scout:
        console.print("\n[yellow]Dry run — no task files written.[/]")
        return

    # Generate task files
    out_dir = output_dir or (repo_path / ".glitchlab" / "tasks" / "queue")
    console.print(f"\n[bold cyan]SCOUT[/] [dim]Generating prioritized task files → {out_dir}[/]")

    config = load_config(repo_path)
    router = Router(config)

    # In scout mode, we let the LLM generate tasks even if static findings are empty
    if scout:
        console.print(f"[bold cyan]SCOUT[/] [dim]Activating Scout Brain (Layer 3)...[/]")

    result.findings = findings
    writer = TaskWriter(router, out_dir, dry_run=dry_run)

    with console.status("[bold cyan]Scout Brain is thinking... (this may take 15-30s)[/]", spinner="point"):
        written = writer.write_tasks(result)

    if dry_run and scout:
         console.print("\n[yellow]Dry run — simulated brain output (no files written).[/]")
         return

    if written:
        console.print(Panel(
            "\n".join(f"  {p.name}" for p in written),
            title=f"[bold green]{len(written)} task files written[/]",
            border_style="green",
        ))

    console.print(f"\n[bold]Next steps:[/]")
    console.print(f"  1. Run batch:        [cyan]glitchlab batch --repo {repo_path} --tasks-dir {out_dir}[/]")
    console.print(f"  2. Monitor:          [cyan]glitchlab history --repo {repo_path} --stats[/]")

@app.command()
def doctor():
    from glitchlab.registry import AGENT_REGISTRY
    from glitchlab.step_handlers import STEP_HANDLERS
    from glitchlab.config_loader import load_config
    from rich.console import Console
    from rich.table import Table
    import sys
    config = load_config()
    console = Console()
    table = Table('Role', 'Registry', 'Handler')
    failed = False
    for step in config.pipeline:
        role = step.agent_role
        in_reg = role in AGENT_REGISTRY
        in_hand = role in STEP_HANDLERS
        if not in_reg or not in_hand: failed = True
        table.add_row(role, '[green]✓[/green]' if in_reg else '[red]✗[/red]', '[green]✓[/green]' if in_hand else '[red]✗[/red]')
    console.print(table)
    if failed: sys.exit(1)

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