# ADR-0007: Controller Decomposition and Minor Version Bump (v4.4.0)

## Status
Accepted

## Context
The GLITCHLAB controller (v3) had grown into a monolithic structure that was becoming difficult to maintain and extend. To improve modularity, testability, and separation of concerns, a significant refactoring was required to decompose the controller into specialized modules.

## Decision
We have decomposed the core controller logic into the following modules:
- `run_context`: Manages the state and context of a task execution.
- `step_handlers`: Contains logic for executing individual steps in a plan.
- `agent_runners`: Handles the orchestration and invocation of different agent types.
- `lifecycle`: Manages the overall lifecycle of the GLITCHLAB process (startup, shutdown, banner).
- `events`: Centralizes event handling and dispatching.

As this represents a significant architectural improvement and a change in internal structure (though maintaining public CLI compatibility), we are bumping the version from 4.3.1 to 4.4.0.

## Consequences
- **Improved Maintainability**: Smaller, focused modules are easier to understand and modify.
- **Enhanced Testability**: Individual components can be unit tested in isolation.
- **Version Alignment**: The 4.4.0 version clearly marks the transition to the decomposed architecture.
- **Internal API Changes**: Developers working on GLITCHLAB internals will need to adapt to the new module structure.
