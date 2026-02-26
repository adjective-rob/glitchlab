from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal

class TaskStepState(BaseModel):
    """Tracks the status of a specific planned step."""
    step_number: int
    status: Literal["pending", "completed", "failed", "skipped"] = "pending"
    outcome: str = ""

class TaskState(BaseModel):
    """The 'Working Memory' for the current agentic loop."""
    task_id: str
    goal: str
    files_in_scope: list[str] = Field(default_factory=list)
    completed_steps: list[TaskStepState] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks_identified: list[str] = Field(default_factory=list)
    current_diff_summary: str = ""
    last_agent_role: str = ""
    
    def to_json(self) -> str:
        return self.model_dump_json(indent=2)