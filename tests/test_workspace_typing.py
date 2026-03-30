import inspect
from pathlib import Path
from glitchlab.workspace import Workspace
from glitchlab.workspace.tools import ToolExecutor


def test_init_return_type_hints():
    """Workspace and ToolExecutor __init__ methods annotate a None return."""
    workspace_init = inspect.signature(Workspace.__init__)
    assert workspace_init.return_annotation is None, "Workspace.__init__ should have '-> None' return type hint"

    tool_executor_init = inspect.signature(ToolExecutor.__init__)
    assert tool_executor_init.return_annotation is None, "ToolExecutor.__init__ should have '-> None' return type hint"


def test_workspace_no_duplicate_rebase():
    """Workspace exposes exactly one rebase method."""
    rebase_methods = [name for name, _ in inspect.getmembers(Workspace, predicate=inspect.isfunction) if name == 'rebase']
    assert len(rebase_methods) == 1, "Workspace should only have one 'rebase' method"
