"""Tests for the agent registry (glitchlab.registry)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from glitchlab.agents.archivist import ArchivistAgent
from glitchlab.agents.debugger import DebuggerAgent
from glitchlab.agents.implementer import ImplementerAgent
from glitchlab.agents.planner import PlannerAgent
from glitchlab.agents.release import ReleaseAgent
from glitchlab.agents.security import SecurityAgent
from glitchlab.agents.testgen import TestGenAgent
from glitchlab.registry import AGENT_REGISTRY, get_agent


EXPECTED_ROLES: dict[str, type] = {
    "planner": PlannerAgent,
    "implementer": ImplementerAgent,
    "debugger": DebuggerAgent,
    "security": SecurityAgent,
    "release": ReleaseAgent,
    "archivist": ArchivistAgent,
    "testgen": TestGenAgent,
}


@pytest.fixture()
def mock_router() -> MagicMock:
    """Provide a router double for registry agent construction tests."""
    return MagicMock()


@pytest.mark.parametrize("role,expected_cls", list(EXPECTED_ROLES.items()))
def test_get_agent_returns_correct_class(
    role: str, expected_cls: type, mock_router: MagicMock
) -> None:
    """get_agent returns an instance of the correct class for each known role."""
    agent = get_agent(role, mock_router)
    assert isinstance(agent, expected_cls)


def test_get_agent_unknown_role_raises(mock_router: MagicMock) -> None:
    """get_agent raises ValueError for an unknown role."""
    with pytest.raises(ValueError, match="Unknown agent role: nonexistent"):
        get_agent("nonexistent", mock_router)


def test_registry_has_all_default_agents() -> None:
    """AGENT_REGISTRY contains exactly the 7 expected agent roles."""
    assert set(AGENT_REGISTRY.keys()) == set(EXPECTED_ROLES.keys())
    assert len(AGENT_REGISTRY) == 7
