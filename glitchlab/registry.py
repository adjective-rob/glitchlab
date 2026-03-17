"""Centralised agent registry for GLITCHLAB.

Maps role names to agent classes and provides a factory function for
instantiating agents by role.
"""

from __future__ import annotations

from glitchlab.agents import BaseAgent
from glitchlab.agents.archivist import ArchivistAgent
from glitchlab.agents.debugger import DebuggerAgent
from glitchlab.agents.implementer import ImplementerAgent
from glitchlab.agents.planner import PlannerAgent
from glitchlab.agents.release import ReleaseAgent
from glitchlab.agents.security import SecurityAgent
from glitchlab.agents.testgen import TestGenAgent
from glitchlab.router import Router

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "planner": PlannerAgent,
    "implementer": ImplementerAgent,
    "debugger": DebuggerAgent,
    "security": SecurityAgent,
    "release": ReleaseAgent,
    "archivist": ArchivistAgent,
    "testgen": TestGenAgent,
}


def get_agent(role: str, router: Router) -> BaseAgent:
    """Instantiate an agent by role name.

    Raises ``ValueError`` if *role* is not present in the registry.
    """
    cls = AGENT_REGISTRY.get(role)
    if not cls:
        raise ValueError(f"Unknown agent role: {role}")
    return cls(router)
