"""
GLITCHLAB Run Context — Per-run shared state bundle.

Replaces the pattern of threading 8+ individual parameters (scope, repo_index,
prelude, history, workspace, state, config, run_id, etc.) through every runner
and handler. A single RunContext is created at the start of each run and passed
to all pipeline components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from glitchlab.agents import BaseAgent
from glitchlab.config_loader import GlitchLabConfig
from glitchlab.task_state import TaskState
from glitchlab.workspace.tools import ToolExecutor


@dataclass
class RunContext:
    """Immutable-ish bag of per-run state shared across pipeline components."""

    # Identity
    run_id: str
    repo_path: Path
    config: GlitchLabConfig

    # Workspace
    ws_path: Path
    workspace: Any  # glitchlab.workspace.Workspace
    tools: ToolExecutor

    # Structured state (mutated by handlers)
    state: TaskState

    # Agents
    agents: dict[str, BaseAgent]

    # Infrastructure
    router: Any           # glitchlab.router.Router
    boundary: Any         # glitchlab.governance.BoundaryEnforcer
    scope: Any            # glitchlab.scope.ScopeResolver
    repo_index: Any       # glitchlab.indexer.RepoIndex
    prelude: Any          # glitchlab.prelude.PreludeContext
    history: Any          # glitchlab.history.TaskHistory

    # Run options
    allow_core: bool = False
    auto_approve: bool = False
    test_command: str | None = None