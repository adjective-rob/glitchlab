"""
GLITCHLAB Controller — The Brainstem (v3)

The most important piece. It is NOT smart. It is deterministic.

v3 Architecture: Decomposed into:
  - run_context.py    — RunContext dataclass (per-run shared state)
  - step_handlers.py  — Per-agent post-processing (registry pattern)
  - agent_runners.py  — Context builders + agent invocations
  - lifecycle.py      — Startup, finalize, PR, session entry

This file is the thin orchestration shell that wires them together.

Responsibilities:
  - Pull next task
  - Create RunContext
  - Iterate pipeline steps (dispatch to handlers)
  - Enforce stop conditions
  - Delegate everything else

It never writes code. It only coordinates.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console

from glitchlab.agents import AgentResult, BaseAgent
from glitchlab.agent_runners import (
    run_archivist,
    run_fix_loop,
    run_implementer,
    run_planner,
    run_release,
    run_security,
    run_testgen,
)
from glitchlab.config_loader import GlitchLabConfig, PipelineStep, load_config
from glitchlab.controller_utils import pre_task_git_fetch
from glitchlab.event_bus import bus
from glitchlab.governance import BoundaryEnforcer
from glitchlab.history import TaskHistory
from glitchlab.lifecycle import (
    check_repo_clean,
    finalize,
    post_run,
    print_banner,
    startup,
    write_session_entry,
)
from glitchlab.prelude import PreludeContext
from glitchlab.registry import AGENT_REGISTRY, get_agent
from glitchlab.router import BudgetExceededError, Router
from glitchlab.run_context import RunContext
from glitchlab.step_handlers import (
    STEP_HANDLERS,
    HandlerSignal,
    PipelineState,
)
from glitchlab.task import Task
from glitchlab.task_state import TaskState, StepState  # re-export

console = Console()


# ---------------------------------------------------------------------------
# Controller (v3)
# ---------------------------------------------------------------------------

class Controller:
    """
    The GLITCHLAB brainstem (v3).

    Thin orchestration shell. All logic lives in:
      - step_handlers.py  (per-agent post-processing)
      - agent_runners.py  (context building + agent invocation)
      - lifecycle.py      (startup, finalize, PR, session)

    Pipeline: Configured via config.yaml pipeline[] steps.
    """

    def __init__(
        self,
        repo_path: Path,
        config: GlitchLabConfig | None = None,
        allow_core: bool = False,
        auto_approve: bool = False,
        surgical: bool = False,
        test_command: str | None = None,
    ):
        self.repo_path = repo_path.resolve()
        self.config = config or load_config(repo_path)
        self.allow_core = allow_core
        self.auto_approve = auto_approve
        self.surgical = surgical
        self.test_command = test_command

        # Core components
        self.router = Router(self.config)
        self.boundary = BoundaryEnforcer(self.config.boundaries.protected_paths)

        # Agents — instantiated from the central registry
        self.agents: dict[str, BaseAgent] = {
            role: get_agent(role, self.router)
            for role in AGENT_REGISTRY
        }

        # History tracking
        self._history = TaskHistory(self.repo_path)

        # Prelude — available as tool context, NOT global prefix
        self._prelude = PreludeContext(self.repo_path)

    # -----------------------------------------------------------------------
    # Main Entry Point
    # -----------------------------------------------------------------------

    def run(self, task: Task) -> dict[str, Any]:
        """Execute the full agent pipeline for a task."""

        # Generate Session Identity for Zephyr
        run_id = str(uuid.uuid4())
        bus.emit(
            event_type="run.started",
            payload={"task_id": task.task_id, "objective": task.objective},
            run_id=run_id,
        )

        # Ensure we plan against the most recent code.
        pre_task_git_fetch(self.repo_path)
        check_repo_clean(self.repo_path)

        # Initialize structured task state
        state = TaskState(
            task_id=task.task_id,
            objective=task.objective,
            mode=task.mode or "evolution",
            risk_level=task.risk_level,
        )

        result: dict[str, Any] = {
            "task_id": task.task_id,
            "status": "pending",
            "events": [],
        }

        print_banner(task)

        # Quality gate: check for ambiguous objectives, inject constraints
        from glitchlab.task_quality import get_quality_constraints
        quality_constraints = get_quality_constraints(task.objective)
        if quality_constraints:
            task.constraints = list(task.constraints or []) + quality_constraints

        # Build RunContext — the shared state bundle for all pipeline components
        ctx = RunContext(
            run_id=run_id,
            repo_path=self.repo_path,
            config=self.config,
            ws_path=self.repo_path,  # placeholder, startup() sets the real path
            workspace=None,
            tools=None,
            state=state,
            agents=self.agents,
            router=self.router,
            boundary=self.boundary,
            scope=None,
            repo_index=None,
            prelude=self._prelude,
            history=self._history,
            allow_core=self.allow_core,
            auto_approve=self.auto_approve,
            surgical=self.surgical,
            test_command=self.test_command,
        )

        try:
            failure_context = startup(ctx, task)

            # Surgical mode: from CLI flag OR from task auto-detection
            if self.surgical or getattr(task, 'surgical', False):
                ctx.surgical = True
                surgical_config = load_config(self.repo_path, profile="surgical")
                ctx.config.pipeline = surgical_config.pipeline
                ctx.config.limits.max_fix_attempts = 1
                if getattr(task, 'surgical', False) and not self.surgical:
                    logger.info(
                        f"[CONTROLLER] Auto-surgical: task '{task.task_id}' detected as trivial. "
                        f"Skipping planner/security/release."
                    )

            ps = self._execute_pipeline(ctx, task, failure_context, result)

            result = finalize(
                ctx, task,
                ps.plan, ps.impl, ps.rel, ps.sec,
                ps.is_doc_only, ps.is_fast_mode,
                ps.result, ps.pipeline_halted,
            )

        except BudgetExceededError as e:
            console.print(f"[red]💸 Budget exceeded: {e}[/]")
            result["status"] = "budget_exceeded"
        except KeyboardInterrupt:
            console.print("\n[yellow]⚡ Interrupted by human.[/]")
            result["status"] = "interrupted"
        except Exception as e:
            logger.exception("Controller error")
            console.print(f"[red]💥 Error: {e}[/]")
            result["status"] = "error"
            result["error"] = str(e)
        finally:
            if ctx.workspace:
                try:
                    ctx.workspace.cleanup()
                except Exception:
                    pass
            self._history.record(result)
            write_session_entry(ctx, task, result)

        result = post_run(ctx, task, result)
        return result

    # -----------------------------------------------------------------------
    # Pipeline Execution
    # -----------------------------------------------------------------------

    def _execute_pipeline(
        self,
        ctx: RunContext,
        task: Task,
        failure_context: str,
        result: dict,
    ) -> PipelineState:
        """Run the dynamic pipeline. Returns PipelineState with all inter-step data."""
        ps = PipelineState(result=result)

        pipeline = ctx.config.pipeline if ctx.surgical else self.config.pipeline

        for step in pipeline:
            if ps.pipeline_halted:
                break

            step_result = self._run_pipeline_step(
                ctx, step, task,
                failure_context=failure_context,
                ps=ps,
            )

            if step_result.payload.get("skipped"):
                # Debugger skip still needs state updates (test phase marking)
                if step.agent_role == "debugger":
                    ctx.state.test_passing = True
                    ctx.state.mark_phase("test")
                    ctx.state.persist(ctx.ws_path)
                continue

            # Dispatch to registered step handler for post-processing
            handler = STEP_HANDLERS.get(step.agent_role)
            if handler:
                signal = handler(ctx, task, step_result, ps)
                if signal == HandlerSignal.EARLY_RETURN:
                    return ps

            # Generic halt for any required step that errors
            if step_result.status == "error" and step.required:
                ps.result["status"] = f"{step.agent_role}_failed"
                ps.pipeline_halted = True

        return ps

    # -----------------------------------------------------------------------
    # Pipeline Step Dispatcher
    # -----------------------------------------------------------------------

    def _run_pipeline_step(
        self,
        ctx: RunContext,
        step: PipelineStep,
        task: Task,
        *,
        failure_context: str = "",
        ps: PipelineState,
    ) -> AgentResult:
        """Execute a single pipeline step by dispatching to the appropriate runner."""

        bus.emit(
            event_type="pipeline.step_started",
            payload={
                "step_name": step.name,
                "agent_role": step.agent_role,
                "required": step.required,
                "skip_if": step.skip_if,
            },
            agent_id=step.agent_role,
            run_id=ctx.run_id,
        )

        # 1. Check skip_if conditions
        skip_conditions: dict[str, bool] = {
            "doc_only": ps.is_doc_only,
            "fast_mode": ps.is_fast_mode,
            "no_test_command": not ctx.test_command,
        }
        for condition in step.skip_if:
            if skip_conditions.get(condition, False):
                bus.emit(
                    event_type="pipeline.step_skipped",
                    payload={
                        "step_name": step.name,
                        "agent_role": step.agent_role,
                        "skip_reason": condition,
                    },
                    agent_id=step.agent_role,
                    run_id=ctx.run_id,
                )
                return AgentResult(
                    status="success",
                    agent=step.agent_role,
                    payload={"skipped": True},
                )

        # 2. Dispatch to the appropriate runner
        role = step.agent_role
        result = self._dispatch_runner(ctx, role, task, failure_context, ps)

        bus.emit(
            event_type="pipeline.step_completed",
            payload={
                "step_name": step.name,
                "agent_role": role,
                "status": result.status,
            },
            agent_id=role,
            run_id=ctx.run_id,
        )

        return result

    def _dispatch_runner(
        self,
        ctx: RunContext,
        role: str,
        task: Task,
        failure_context: str,
        ps: PipelineState,
    ) -> AgentResult:
        """Route to the correct agent runner by role."""

        if role == "planner":
            return run_planner(ctx, task, failure_context)

        if role == "implementer":
            if ps.is_doc_only:
                return AgentResult(
                    status="success", agent="implementer",
                    payload={"doc_only": True},
                )
            return run_implementer(ctx, task, ps.plan)

        if role == "testgen":
            run_testgen(ctx, task, ps.is_doc_only)
            return AgentResult(status="success", agent="testgen")

        if role == "debugger":
            test_ok = run_fix_loop(ctx, task, ps.impl)
            return AgentResult(
                status="success", agent="debugger",
                payload={"test_passing": test_ok},
            )

        if role == "security":
            return run_security(ctx, task, ps.is_fast_mode)

        if role == "release":
            return run_release(ctx, task, ps.is_fast_mode)

        if role == "archivist":
            return run_archivist(ctx, task, ps.is_fast_mode)

        raise ValueError(f"Unknown pipeline agent_role: {role!r}")