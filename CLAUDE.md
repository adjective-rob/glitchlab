# CLAUDE.md — GLITCHLAB v4.2.0

## Identity

GLITCHLAB is a local-first, repo-agnostic, multi-agent development engine. The Controller (`controller.py`) is a deterministic orchestrator. It never writes code. Agents are stateless between runs. State lives in the repo via Pydantic models (`TaskState`, `StepState`).

## Critical Rules

1. **No speculative refactors.** Only change what the task explicitly requires. If a file is not named in the task, do not touch it.
2. **No renaming, moving, or deleting files** unless the task specifically asks for it.
3. **No changing function signatures** on public API surfaces (`BaseAgent`, `AgentContext`, `Router`, `Workspace`, `ToolExecutor`, `EventBus`, `BoundaryEnforcer`, `SymbolIndex`, `TaskState`) without explicit approval.
4. **No rewriting files from scratch.** Use targeted edits. If you must rewrite, explain why before doing it.
5. **Run tests after every change:** `python -m pytest tests/`
6. **Run the linter after every change:** `python -m ruff check glitchlab/`
7. **Do not modify `config.yaml` defaults** (routing, limits, allowed_tools, blocked_patterns) unless the task is specifically about configuration.

## Architecture — Do Not Violate

- **Controller is the brainstem.** Pipeline order: Index → Plan → Implement → Test → Debug → Security → Release → PR. Do not reorder or skip phases.
- **Agents inherit from `BaseAgent`** (`glitchlab/agents/__init__.py`). Each agent implements `build_messages()` and `parse_response()`. Do not add logic to `BaseAgent.run()` without discussion.
- **Context-Router pattern.** `TaskState.to_agent_summary(for_agent)` controls what each agent sees. Agents pull context through tools, they are not pushed full state. Do not expand agent summaries without justification.
- **EventBus is a global singleton** (`glitchlab/event_bus.py`). Events carry `run_id`, `action_id`, `metadata`. All agent actions and tool executions emit events. Do not bypass the bus for observability.
- **Workspace isolation uses git worktrees.** Agents never write to `main`. Do not change `Workspace.create()`, `Workspace.cleanup()`, or the worktree lifecycle without explicit ask.
- **ToolExecutor enforces allowlist + blocklist.** Commands are prefix-matched against `allowed_tools` and rejected if they match `blocked_patterns`. Do not weaken these checks.
- **BoundaryEnforcer** gates protected paths behind `--allow-core`. Do not remove or soften boundary checks.
- **Router** wraps LiteLLM with budget tracking and retry logic. Agents never know which vendor backs them. Do not leak model names into agent prompts.

## File Ownership — Treat as Load-Bearing

These files have high coupling. Changes here ripple across the entire engine. Read downstream consumers before editing.

| File | Why it's dangerous |
|---|---|
| `controller.py` | 2100 lines. Orchestrates everything. Houses `TaskState` (the canonical one). |
| `agents/__init__.py` | `BaseAgent` + `AgentContext` — every agent depends on these shapes. |
| `workspace/__init__.py` | Git worktree lifecycle. Breakage = data loss or orphaned branches. |
| `workspace/tools.py` | `ToolExecutor` — the security boundary for all shell execution. |
| `router.py` | Model routing + budget tracking. Breakage = runaway costs or silent failures. |
| `event_bus.py` | Global singleton. Schema changes break all event consumers. |
| `governance/__init__.py` | `BoundaryEnforcer`. Weakening this removes safety rails. |
| `config.yaml` | Runtime defaults. Changing blocked_patterns or allowed_tools = security surface change. |

## Dual TaskState Warning

There are two `TaskState` definitions: one in `controller.py` (canonical, used at runtime) and one in `state.py` (structural duplicate). The controller's version is authoritative. If you modify TaskState fields, update `controller.py`. Do not assume `state.py` is the source of truth.

## Conventions

- **Python 3.11+.** Type hints everywhere. Pydantic v2 models for structured data.
- **Ruff** for linting. Line length 100. Target `py311`.
- **Loguru** for logging. Use `logger.info`, `logger.debug`, `logger.warning`. No `print()`.
- **Rich** for CLI output. Console output goes through `rich.console.Console`.
- **Tests live in `tests/`.** Pytest. No test file outside that directory.
- **ADRs live in `docs/adr/`.** If a change alters architecture, write an ADR.
- **Imports:** Absolute imports from `glitchlab.*`. No relative imports.

## Testing Requirements

- Every new public function gets a test.
- Do not delete existing tests to make a change pass.
- If a test fails after your change, fix the code — not the test — unless the test was wrong.
- The existing test suite is thin. That makes it more important, not less.

## What "Optimize" Means Here

When asked to optimize, the goal is:
- Reduce token consumption in agent prompts (context routing, not context hoarding).
- Reduce redundant I/O or subprocess calls.
- Tighten type safety or error handling.
- Improve clarity of agent system prompts.
- **Not:** rewrite modules, introduce new abstractions, add dependencies, or restructure the pipeline.

## Before You Start Any Task

1. Read the specific files involved. Use `find_references` or `grep` to understand call sites.
2. Identify which tests cover the code you are changing.
3. State your plan (what changes, which files, why) before writing code.
4. Make the smallest change that satisfies the task.
5. Run tests. Run linter. Confirm green before declaring done.