"""
Configuration loader for GLITCHLAB.
Merges defaults with per-repo .glitchlab/config.yaml overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class RoutingConfig(BaseModel):
    planner: str = "gemini/gemini-3.1-pro-preview"
    implementer: str = "gemini/gemini-3.1-pro-preview"
    debugger: str = "gemini/gemini-3.1-pro-preview"
    security: str = "gemini/gemini-3-flash-preview"
    release: str = "gemini/gemini-3-flash-preview"
    archivist: str = "gemini/gemini-3-flash-preview"


class LimitsConfig(BaseModel):
    max_fix_attempts: int = 4
    max_tokens_per_task: int = 150_000
    max_dollars_per_task: float = 10.0
    require_plan_review: bool = True
    require_pr_review: bool = True


class InterventionConfig(BaseModel):
    pause_after_plan: bool = True
    pause_before_pr: bool = True
    pause_on_core_change: bool = True
    pause_on_budget_exceeded: bool = True


class WorkspaceConfig(BaseModel):
    worktree_dir: str = ".glitchlab/worktrees"
    task_dir: str = ".glitchlab/tasks"
    log_dir: str = ".glitchlab/logs"


class BoundaryConfig(BaseModel):
    protected_paths: list[str] = Field(default_factory=list)
    allow_test_modifications: bool = False


class ContextConfig(BaseModel):
    brain: str = "~/.glitchlab/brain"
    min_version: str = "1.2.0"


class GlitchLabConfig(BaseModel):
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    intervention: InterventionConfig = Field(default_factory=InterventionConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    allowed_tools: list[str] = Field(default_factory=list)
    blocked_patterns: list[str] = Field(default_factory=list)
    boundaries: BoundaryConfig = Field(default_factory=BoundaryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(repo_path: Path | None = None) -> GlitchLabConfig:
    """
    Load config by merging:
      1. Built-in defaults (glitchlab/config.yaml)
      2. Repo-level overrides (<repo>/.glitchlab/config.yaml)
      3. Environment variable overrides
    """
    # 1. Built-in defaults
    with open(_DEFAULT_CONFIG_PATH, "r") as f:
        base: dict[str, Any] = yaml.safe_load(f) or {}

    # 2. Repo overrides
    if repo_path:
        repo_config = repo_path / ".glitchlab" / "config.yaml"
        if repo_config.exists():
            with open(repo_config, "r") as f:
                overrides: dict[str, Any] = yaml.safe_load(f) or {}
            base = _deep_merge(base, overrides)

    # 3. Env overrides for API keys (informational — LiteLLM reads these directly)
    # We just validate they exist.
    config = GlitchLabConfig(**base)
    return config


def validate_api_keys() -> dict[str, bool]:
    """Check which API keys are available."""
    return {
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "OPENAI_API_KEY":    bool(os.environ.get("OPENAI_API_KEY")),
        "GOOGLE_API_KEY":    bool(os.environ.get("GOOGLE_API_KEY")),
        "GEMINI_API_KEY":    bool(os.environ.get("GEMINI_API_KEY")),
    }