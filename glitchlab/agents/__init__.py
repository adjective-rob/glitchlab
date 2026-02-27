"""
GLITCHLAB Agent Roster

Each agent is:
  - A system prompt
  - A structured input template
  - A constrained output schema

Agents are stateless between runs. State lives in the repo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from glitchlab.router import Router, RouterResponse


class AgentContext(BaseModel):
    """Shared context passed to every agent invocation."""
    task_id: str
    objective: str
    repo_path: str
    working_dir: str
    constraints: list[str] = []
    acceptance_criteria: list[str] = []
    risk_level: str = "low"
    file_context: dict[str, str] = {}  # filename → content snippets
    previous_output: dict[str, Any] = {}  # output from prior agent in chain
    extra: dict[str, Any] = {}


class BaseAgent(ABC):
    """
    Base class for all GLITCHLAB agents.

    Subclasses define:
      - role: str — maps to router model
      - system_prompt: str — agent personality + constraints
      - build_messages() — constructs the chat messages
      - parse_response() — extracts structured output
    """

    role: str = "unknown"
    system_prompt: str = "You are a helpful assistant."

    def __init__(self, router: Router):
        self.router = router

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Execute the agent: build messages → call model → parse."""
        messages = self.build_messages(context)
        response = self.router.complete(
            role=self.role,
            messages=messages,
            **kwargs,
        )
        return self.parse_response(response, context)

    @abstractmethod
    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        """Build the message list for the LLM call."""
        ...

    @abstractmethod
    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        """Parse the LLM response into structured output."""
        ...

    def _system_msg(self) -> dict[str, str]:
        return {"role": "system", "content": self.system_prompt}

    def _user_msg(self, content: str) -> dict[str, str]:
        return {"role": "user", "content": content}
