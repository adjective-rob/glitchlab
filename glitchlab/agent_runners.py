"""
GLITCHLAB Agent Runners — Context builders + agent invocations.

Each function builds an AgentContext, runs the agent, and returns a result.
They are stateless helpers called by the Controller's pipeline dispatcher.

Merged from:
  - Original runners.py (security, release, archivist, delegated)
  - Controller._run_planner, _run_implementer, _run_fix_loop, _run_testgen, _retry_patch
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console

from glitchlab.agents import AgentContext, BaseAgent, AgentResult
from glitchlab.controller_utils import attest_controller_action
from glitchlab.display import print_plan
from glitchlab.doc_inserter import insert_doc_comments
from glitchlab.events import emit_event
from glitchlab.history import extract_patterns_from_messages
from glitchlab.run_context import RunContext
from glitchlab.symbols import SymbolIndex
from glitchlab.task import Task
from glitchlab.task_state import TaskState
from glitchlab.workspace.tools import ToolViolationError

console = Console()


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def run_planner(ctx: RunContext, task: Task, failure_context: str = "") -> AgentResult:
    console.print("\n[bold magenta]🧠 [ZAP] Planning...[/]")

    # Planner gets: repo file map + task + failure history
    # NO global Prelude dump. Prelude constraints already merged into task.
    objective_parts = []

    repo_index_context = ctx.repo_index.to_agent_context(max_files=200) if ctx.repo_index else ""
    if repo_index_context:
        objective_parts.append(repo_index_context)

    if failure_context:
        objective_parts.append(failure_context)

    objective_parts.append(f"TASK:\n{task.objective}")

    objective = "\n\n---\n\n".join(objective_parts)

    symbol_index = SymbolIndex(ctx.ws_path)

    context = AgentContext(
        task_id=task.task_id,
        run_id=ctx.run_id,
        objective=objective,
        repo_path=str(ctx.repo_path),
        working_dir=str(ctx.ws_path),
        constraints=task.constraints,
        acceptance_criteria=task.acceptance_criteria,
        risk_level=task.risk_level,
        extra={
            "prelude": ctx.prelude,
            "symbol_index": symbol_index,
        },
    )

    raw = ctx.agents["planner"].run(context)
    emit_event(ctx, "plan_created", {
        "steps": len(raw.get("steps", [])),
        "risk": raw.get("risk_level"),
    })

    print_plan(raw)

    if ctx.config.intervention.pause_after_plan and not ctx.auto_approve:
        from rich.prompt import Confirm
        if not Confirm.ask("[bold]Approve plan?[/]"):
            raw["_aborted"] = True
            raw["parse_error"] = True
            return AgentResult.from_raw(raw)

    return AgentResult.from_raw(raw)


# ---------------------------------------------------------------------------
# Implementer
# ---------------------------------------------------------------------------

def run_implementer(ctx: RunContext, task: Task, plan: dict) -> AgentResult:
    console.print("\n[bold blue]🔧 [PATCH] Implementing...[/]")

    # AST Layer Initialization
    symbol_index = SymbolIndex(ctx.ws_path)

    # ScopeResolver computes context from actual imports
    file_context = ctx.scope.resolve_for_files(
        plan.get("files_likely_affected", []),
        include_deps=True,
        signatures_only=True,
    )

    # Keep the user's task constraints, but DROP the JSON formatting constraints
    impl_constraints = list(task.constraints)

    # Add plan-level do_not_touch as constraints
    plan_dnt = plan.get("do_not_touch", [])
    if plan_dnt:
        impl_constraints.append(f"DO NOT TOUCH these files/symbols: {', '.join(plan_dnt)}")
    
    # Rewrite heuristic — prefer write_file over surgical edits for small files
    for f in plan.get("files_likely_affected", []):
        fpath = ctx.ws_path / f
        if fpath.exists() and fpath.stat().st_size > 0:
            line_count = len(fpath.read_text().splitlines())
            if line_count < 200:
                impl_constraints.append(
                    f"File {f} is {line_count} lines — use write_file to rewrite it entirely instead of multiple replace_in_file calls."
                )

    # Multi-edit heuristic — if 3+ plan steps touch the same file, rewrite instead
    from collections import Counter
    file_touch_counts = Counter(
        f for step in plan.get("steps", []) for f in step.get("files", [])
    )
    for f, count in file_touch_counts.items():
        if count >= 3:
            impl_constraints.append(
                f"File {f} is touched by {count} plan steps — use write_file to rewrite it entirely instead of {count} separate replace_in_file calls."
            )

    # Memory Injection
    heuristics = ctx.history.build_heuristics(plan.get("files_likely_affected", []))

    # Brain hints (persistent cross-run codebase memory)
    brain_hints = ""
    try:
        from glitchlab.brain_writer import read_brain_hints
        brain_dir = Path(ctx.config.context.brain).expanduser()
        brain_hints = read_brain_hints(
            brain_dir, ctx.repo_path.name, plan.get("files_likely_affected", [])
        )
    except Exception:
        pass
    if brain_hints:
        heuristics = (heuristics + "\n\n" + brain_hints).strip() if heuristics else brain_hints

    # Pass structured task state AND the tool executor
    context = AgentContext(
        task_id=task.task_id,
        run_id=ctx.run_id,
        objective=task.objective,
        repo_path=str(ctx.repo_path),
        working_dir=str(ctx.ws_path),
        constraints=impl_constraints,
        acceptance_criteria=task.acceptance_criteria,
        file_context=file_context,
        previous_output=ctx.state.to_agent_summary("implementer"),
        extra={
            "tool_executor": ctx.tools,
            "test_command": ctx.test_command,
            "learned_heuristics": heuristics,
            "symbol_index": symbol_index,
            "prelude": ctx.prelude,
            "fast_mode": (
                ctx.state.estimated_complexity in ("trivial", "small")
                or len(ctx.state.files_in_scope) <= 3
            ),
        },
    )

    # The Switchboard Delegation Loop
    while True:
        impl = ctx.agents["implementer"].run(context, max_tokens=12000)

        # Did Patch yield to ask for help?
        if impl.get("_status") == "delegating":
            target = impl.get("colleague", "unknown")
            request = impl.get("request", "No specific request provided.")
            tc_id = impl.get("tc_id")
            tc_name = impl.get("tc_name")

            console.print(f"\n[bold magenta]📞 Patch is tagging in {target.upper()}...[/]")
            console.print(f"  [dim]Request: {request}[/]")

            # 1. Spin up the requested agent
            colleague_response = run_delegated_agent(
                ctx=ctx, target=target, request=request, task=task,
            )

            # 2. Inject the response back into Patch's memory
            context.extra["_resume_messages"] = impl.get("_messages", [])
            context.extra["_resume_messages"].append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": tc_name,
                "content": f"Colleague {target.upper()} responded:\n{colleague_response}"
            })

            console.print("[bold blue]🔄 Resuming Patch...[/]")
            continue

        break  # Exit loop when Patch successfully calls `done` (or hits a hard error)

    # Memory Extraction
    messages = impl.get("_messages", [])
    impl_result = AgentResult.from_raw(impl)
    if messages:
        outcome = "fail" if impl_result.status == "error" else "pass"
        patterns = extract_patterns_from_messages(messages, outcome)
        if patterns:
            ctx.history.record_patterns(task.task_id, patterns)
        # Stash messages for brain writer in post_run (non-underscore key survives from_raw filter)
        impl_result.payload["impl_messages"] = messages

    # For doc-comment tasks, use surgical insertion
    for change in impl_result.payload.get("changes", []):
        if change.get("action") == "modify" and not change.get("_already_applied"):
            fpath = ctx.ws_path / change["file"]
            if fpath.exists():
                inserted = insert_doc_comments(fpath, ctx.router)
                if inserted:
                    change["_doc_inserted"] = True
                    change["content"] = None

    emit_event(ctx, "implementation_created", {
        "changes": len(impl_result.payload.get("changes", [])),
        "tests": len(impl_result.payload.get("tests_added", [])),
    })

    return impl_result


# ---------------------------------------------------------------------------
# Testgen (Shield)
# ---------------------------------------------------------------------------

def run_testgen(ctx: RunContext, task: Task, is_doc_only: bool) -> None:
    """Run the Shield agent to generate a regression test if none exists."""
    if is_doc_only:
        return

    # Check if tests already created by the implementer
    existing_tests = ctx.state.tests_added[:]
    for f in ctx.state.files_created + ctx.state.files_modified:
        if "test_" in f.lower() or "_test" in f.lower() or f.startswith("tests/"):
            existing_tests.append(f)

    if existing_tests:
        console.print(f"  [dim]Tests already exist/created: {existing_tests[0]}. Skipping Shield.[/]")
        return

    console.print("\n[bold green]🛡️ [SHIELD] Generating regression test...[/]")

    # Resolve actual written code for Shield to analyze
    file_context = ctx.scope.resolve_for_files(
        ctx.state.files_modified,
        include_deps=False,
    )

    context = AgentContext(
        task_id=task.task_id,
        run_id=ctx.run_id,
        objective=task.objective,
        repo_path=str(ctx.repo_path),
        working_dir=str(ctx.ws_path),
        previous_output=ctx.state.to_agent_summary("testgen"),
        file_context=file_context,
        extra={"test_command": ctx.test_command},
    )

    raw = ctx.agents["testgen"].run(context)
    tg_result = AgentResult.from_raw(raw)

    if tg_result.status == "error" or not tg_result.payload.get("test_file"):
        console.print("  [yellow]Shield failed to generate a valid test. Continuing.[/]")
        return

    test_file = tg_result.payload["test_file"]
    content = tg_result.payload["content"]
    desc = tg_result.payload["description"]

    try:
        fpath = ctx.ws_path / test_file
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content)
        ctx.state.tests_added.append(test_file)
        emit_event(ctx, "testgen_created", {"file": test_file, "description": desc}, agent_id="testgen")
        console.print(f"  [cyan]TESTGEN {test_file}[/]")
        console.print(f"  [dim]Generated: {desc}[/]")
        attest_controller_action(f"TESTGEN {test_file}", ctx.run_id)
    except Exception as e:
        console.print(f"  [red]Failed to write test file: {e}[/]")


# ---------------------------------------------------------------------------
# Fix Loop (Debugger)
# ---------------------------------------------------------------------------

def run_fix_loop(ctx: RunContext, task: Task, impl: dict) -> bool:
    """
    Run test → debug → fix loop (v3.0).
    Debugger is now agentic and manages its own tool-loop to investigate and fix.
    """
    max_attempts = ctx.config.limits.max_fix_attempts

    for attempt in range(1, max_attempts + 1):
        console.print(f"\n[bold]🧪 Test run {attempt}/{max_attempts}...[/]")

        try:
            result = ctx.tools.execute(ctx.test_command)
        except ToolViolationError as e:
            console.print(f"[red]Tool violation: {e}[/]")
            return False

        if result.success:
            console.print("[green]✅ Tests pass![/]")
            emit_event(ctx, "tests_passed", {"attempt": attempt}, agent_id="debugger")
            return True

        error_output = result.stderr or result.stdout
        console.print(f"[red]❌ Tests failed (attempt {attempt})[/]")
        emit_event(ctx, "tests_failed", {"attempt": attempt}, agent_id="debugger")

        if attempt >= max_attempts:
            break

        # Invoke Debugger (Agentic Loop)
        console.print(f"\n[bold yellow]🐛 [REROUTE] Debugging (attempt {attempt})...[/]")

        context = AgentContext(
            task_id=task.task_id,
            run_id=ctx.run_id,
            objective=task.objective,
            repo_path=str(ctx.repo_path),
            working_dir=str(ctx.ws_path),
            previous_output=ctx.state.to_agent_summary("debugger"),
            extra={
                "error_output": (error_output or "")[-3000:],
                "test_command": ctx.test_command,
                "tool_executor": ctx.tools,
                "prelude": ctx.prelude,
                "repo_index": ctx.repo_index,
                "fast_mode": (
                    len(ctx.state.files_in_scope) <= 3
                    and ctx.state.estimated_complexity in ("trivial", "small")
                ),
            },
        )

        # Debugger now runs its own 10-step loop internally
        raw_debug = ctx.agents["debugger"].run(context)
        debug_result = AgentResult.from_raw(raw_debug)

        emit_event(ctx, "debug_completed", {
            "attempt": attempt,
            "status": debug_result.status,
            "diagnosis": debug_result.payload.get("diagnosis", ""),
            "root_cause": debug_result.payload.get("root_cause", ""),
            "files_fixed": [c.get("file") for c in debug_result.payload.get("fix", {}).get("changes", []) if c.get("file")],
        }, agent_id="debugger")

        # Record debug turn for TaskHistory
        ctx.state.previous_fixes.append(debug_result.payload)
        ctx.state.last_error = debug_result.payload.get("diagnosis", "Unknown error")
        ctx.state.debug_attempts = attempt

        # Record failure memory
        fix_changes = debug_result.payload.get("fix", {}).get("changes", [])
        for change in fix_changes:
            if change.get("file"):
                ctx.history.record_failure_detail(
                    task_id=task.task_id,
                    file_modified=change["file"],
                    error_type=ctx.state.last_error,
                    resolution=debug_result.payload.get("root_cause", "Fixed in debug loop"),
                )

        # Sync TaskState with files written by the debugger's tools
        fix_changes = debug_result.payload.get("fix", {}).get("changes", [])
        for change in fix_changes:
            f = change.get("file")
            if f:
                if change.get("action") == "create" and f not in ctx.state.files_created:
                    ctx.state.files_created.append(f)
                elif f not in ctx.state.files_modified:
                    ctx.state.files_modified.append(f)

        if debug_result.status == "error":
            console.print("[yellow]⚠ Debugger failed to conclude. Retrying loop...[/]")
            continue

        if not debug_result.payload.get("should_retry", False):
            console.print("[yellow]Debugger suggests abandoning fix.[/]")
            break

    return False


# ---------------------------------------------------------------------------
# Retry Patch
# ---------------------------------------------------------------------------

def retry_patch(
    ctx: RunContext, task: Task, plan: dict, original_impl: dict, applied_entries: list[str],
) -> AgentResult:
    console.print("[dim]Re-prompting implementer with git error context...[/]")

    error_lines = [
        a.replace("PATCH_ERROR ", "")
        for a in applied_entries
        if a.startswith("PATCH_ERROR")
    ]
    git_error = "\n".join(error_lines)

    file_context = ctx.scope.resolve_for_files(
        plan.get("files_likely_affected", []),
        include_deps=False,
    )

    retry_prompt = f"""
The previous unified diff failed to apply with the following git error:

{git_error}

Regenerate a corrected unified diff that will apply cleanly.
Ensure:
- At least 3 context lines
- Exact whitespace match
- Valid unified diff format
- No unrelated changes
"""

    context = AgentContext(
        task_id=task.task_id,
        run_id=ctx.run_id,
        objective=task.objective + "\n\n" + retry_prompt,
        repo_path=str(ctx.repo_path),
        working_dir=str(ctx.ws_path),
        constraints=task.constraints,
        acceptance_criteria=task.acceptance_criteria,
        file_context=file_context,
        previous_output=ctx.state.to_agent_summary("implementer"),
    )

    raw = ctx.agents["implementer"].run(context, max_tokens=8192)
    return AgentResult.from_raw(raw)


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------

def run_auditor(
    ctx: RunContext, task: Task,
) -> dict:
    """Run the auditor agent to check for performance smells."""
    console.print("\n[bold yellow]🕵️  [AUDITOR] Checking for performance smells...[/]")

    diff = ctx.workspace.diff_full() if ctx.workspace else ""

    context = AgentContext(
        task_id=task.task_id,
        run_id=ctx.run_id,
        objective=task.objective,
        repo_path=str(ctx.repo_path),
        working_dir=str(ctx.ws_path),
        previous_output=ctx.state.to_agent_summary("auditor"),
        extra={"diff": diff},
    )

    return ctx.agents.get("auditor", ctx.agents["planner"]).run(context)


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

def run_security(
    ctx: RunContext, task: Task, is_fast_mode: bool = False,
) -> AgentResult:
    """Run the security agent to scan for vulnerabilities."""
    console.print("\n[bold red]🔒 [FRANKIE] Security scan...[/]")

    diff = ctx.workspace.diff_full() if ctx.workspace else ""

    context = AgentContext(
        task_id=task.task_id,
        run_id=ctx.run_id,
        objective=task.objective,
        repo_path=str(ctx.repo_path),
        working_dir=str(ctx.ws_path),
        previous_output=ctx.state.to_agent_summary("security"),
        extra={
            "diff": diff,
            "protected_paths": ctx.config.boundaries.protected_paths,
            "fast_mode": is_fast_mode,
            "repo_index": ctx.repo_index,
            "prelude": ctx.prelude,
        },
    )

    raw = ctx.agents["security"].run(context, max_steps=10 if is_fast_mode else 30)
    return AgentResult.from_raw(raw)


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------

def run_release(
    ctx: RunContext, task: Task, is_fast_mode: bool = False,
) -> AgentResult:
    """Run the release agent for version bump assessment."""
    console.print("\n[bold cyan]📦 [SEMVER] Release assessment...[/]")

    diff = ctx.workspace.diff_stat() if ctx.workspace else ""

    context = AgentContext(
        task_id=task.task_id,
        run_id=ctx.run_id,
        objective=task.objective,
        repo_path=str(ctx.repo_path),
        working_dir=str(ctx.ws_path),
        previous_output=ctx.state.to_agent_summary("release"),
        extra={
            "diff": diff,
            "fast_mode": is_fast_mode,
        },
    )

    raw = ctx.agents["release"].run(context, max_steps=10 if is_fast_mode else 20)
    return AgentResult.from_raw(raw)


# ---------------------------------------------------------------------------
# Archivist
# ---------------------------------------------------------------------------

def run_archivist(
    ctx: RunContext, task: Task, is_fast_mode: bool = False,
) -> AgentResult:
    """Run Archivist Nova with structured state context."""
    console.print("\n[bold dim]📚 [NOVA] Documenting...[/]")

    existing_docs = []
    for pattern in ["*.md", "docs/**/*.md", "doc/**/*.md"]:
        existing_docs.extend(
            str(p.relative_to(ctx.ws_path))
            for p in ctx.ws_path.glob(pattern)
            if p.is_file() and ".glitchlab" not in str(p)
        )

    context = AgentContext(
        task_id=task.task_id,
        run_id=ctx.run_id,
        objective=task.objective,
        repo_path=str(ctx.repo_path),
        working_dir=str(ctx.ws_path),
        previous_output=ctx.state.to_agent_summary("archivist"),
        extra={
            "existing_docs": existing_docs[:50],
            "fast_mode": is_fast_mode,
            "today": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        },
    )

    raw = ctx.agents["archivist"].run(context)

    if raw is None:
        return AgentResult(
            status="error",
            payload={
                "should_write_adr": False,
                "doc_updates": [],
                "architecture_notes": "Archivist failed.",
            },
        )

    return AgentResult.from_raw(raw)


# ---------------------------------------------------------------------------
# Delegated Agent (mid-flight switchboard)
# ---------------------------------------------------------------------------

def run_delegated_agent(
    ctx: RunContext, target: str, request: str, task: Task,
) -> str:
    """Handle mid-flight delegation requests from the Implementer."""
    sub_context = AgentContext(
        task_id=f"{task.task_id}-delegate-{target}",
        run_id=ctx.run_id,
        objective=(
            f"Your colleague needs your expertise on a specific sub-task:\n\n{request}"
        ),
        repo_path=str(ctx.repo_path),
        working_dir=str(ctx.ws_path),
        extra={
            "tool_executor": ctx.tools,
            "prelude": ctx.prelude,
            "fast_mode": False,
            "repo_index": ctx.repo_index,
        },
    )

    try:
        if target == "security":
            res = ctx.agents["security"].run(sub_context)
            return (
                f"Verdict: {res.get('verdict')}\n"
                f"Summary: {res.get('summary')}\n"
                f"Issues: {res.get('issues', [])}"
            )

        elif target == "debugger":
            sub_context.extra["test_command"] = ctx.test_command
            res = ctx.agents["debugger"].run(sub_context)
            return (
                f"Diagnosis: {res.get('diagnosis')}\n"
                f"Root Cause: {res.get('root_cause')}\n"
                f"Fixes applied: {res.get('fix_summary', 'None')}"
            )

        elif target == "testgen":
            sub_context.extra["test_command"] = ctx.test_command
            res = ctx.agents["testgen"].run(sub_context)
            return (
                f"Test Generated: {res.get('test_file')}\n"
                f"Description: {res.get('description')}"
            )

        elif target == "archivist":
            res = ctx.agents["archivist"].run(sub_context)
            return (
                f"Architecture Notes: {res.get('architecture_notes')}\n"
                f"ADR Written: {res.get('should_write_adr')}"
            )

        else:
            return f"Error: Unknown colleague '{target}'."

    except Exception as e:
        logger.error(f"Delegation to {target} failed: {e}")
        return (
            f"Colleague {target} encountered an error and "
            f"could not complete the request: {e}"
        )