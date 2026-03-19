"""
GLITCHLAB Task State — Structured Working Memory (Layer 3)

Canonical definition of TaskState and StepState.
The Controller owns TaskState instances and persists them per-run.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field


class StepState(BaseModel):
    """Tracks the status of an individual planned step."""
    step_number: int
    description: str = ""
    files: list[str] = Field(default_factory=list)
    action: str = ""
    status: Literal["pending", "completed", "failed", "skipped"] = "pending"
    outcome: str = ""
    do_not_touch: list[str] = Field(default_factory=list)
    code_hint: str = ""


class TaskState(BaseModel):
    """
    Structured working memory that flows between agents.

    Replaces the old pattern of passing raw `previous_output` blobs.
    Each agent reads what it needs and writes its contribution.
    The Controller owns this object and persists it per-run.
    """

    task_id: str
    objective: str
    mode: str = "evolution"
    risk_level: str = "low"

    # Planner output (consumed by Implementer, Debugger, Security)
    plan_steps: list[StepState] = Field(default_factory=list)
    files_in_scope: list[str] = Field(default_factory=list)
    estimated_complexity: str = "medium"
    requires_core_change: bool = False

    # Implementer output (consumed by Debugger, Security, Release)
    files_modified: list[str] = Field(default_factory=list)
    files_created: list[str] = Field(default_factory=list)
    tests_added: list[str] = Field(default_factory=list)
    commit_message: str = ""
    implementation_summary: str = ""

    # Debug loop state
    test_passing: bool = False
    debug_attempts: int = 0
    last_error: str = ""
    previous_fixes: list[dict] = Field(default_factory=list)

    # Security + Release
    security_verdict: str = ""
    version_bump: str = ""
    changelog_entry: str = ""

    # Tracking
    completed_phases: list[str] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)

    AGENT_FIELDS: ClassVar[dict[str, list[str]]] = {
        "planner": ["previous_fixes"],
        "implementer": ["plan_steps", "files_in_scope", "estimated_complexity"],
        "testgen": ["files_modified", "files_created", "implementation_summary"],
        "debugger": ["files_modified", "files_created", "last_error",
                     "debug_attempts", "previous_fixes"],
        "auditor": ["files_modified", "files_created", "implementation_summary"],
        "security": ["files_modified", "files_created", "implementation_summary"],
        "release": ["files_modified", "implementation_summary", "security_verdict"],
        "archivist": ["plan_steps", "files_modified", "implementation_summary",
                      "version_bump"],
    }

    FIELD_CAPS: ClassVar[dict[tuple[str, str], int]] = {
        ("planner", "previous_fixes"): 3,
        ("debugger", "previous_fixes"): 2,
    }

    def mark_phase(self, phase: str) -> None:
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)

    def to_agent_summary(self, for_agent: str) -> dict:
        """
        Return only the fields relevant to a specific agent.
        This is the core of the context-router pattern: agents get
        precisely what they need, not everything.

        Field routing is driven by AGENT_FIELDS (which fields each agent
        sees) and FIELD_CAPS (optional tail-slice limits for list fields).
        New agent roles can be added by extending AGENT_FIELDS.
        """
        base = {
            "task_id": self.task_id,
            "objective": self.objective,
            "mode": self.mode,
            "risk_level": self.risk_level,
        }
        fields = self.AGENT_FIELDS.get(for_agent, [])
        for field_name in fields:
            value = getattr(self, field_name, None)
            cap = self.FIELD_CAPS.get((for_agent, field_name))
            if cap is not None:
                value = value[-cap:] if value else []
            elif isinstance(value, list) and all(
                hasattr(v, "model_dump") for v in value
            ):
                value = [v.model_dump() for v in value]
            base[field_name] = value
        return base

    def persist(self, ws_path: Path) -> None:
        """Write current state to workspace for debugging/auditing."""
        state_dir = ws_path / ".glitchlab"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "task_state.json").write_text(
            self.model_dump_json(indent=2)
        )


class DirtyRepoError(Exception):
    """Raised when the main repository has uncommitted changes."""
    pass
