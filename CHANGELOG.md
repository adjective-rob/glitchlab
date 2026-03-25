# Changelog

## [Unreleased]

## [4.4.0] - 2026-03-25

### Changed
- **Controller Decomposition**: Completed the v3 controller decomposition, refactoring the core logic into specialized modules: `run_context`, `step_handlers`, `agent_runners`, `lifecycle`, and `events`.
- **Version Bump**: Updated version string to 4.4.0 across `glitchlab/__init__.py`, `glitchlab/identity.py`, `pyproject.toml`, `glitchlab/lifecycle.py`, and tests.

### Changed
- **Controller Decomposition**: Completed the v3 controller decomposition, refactoring the core logic into specialized modules: `run_context`, `step_handlers`, `agent_runners`, `lifecycle`, and `events`.
- **Version Bump**: Updated version string to 4.4.0 across `glitchlab/__init__.py`, `glitchlab/identity.py`, `pyproject.toml`, `glitchlab/lifecycle.py`, and tests.

### Added
- **Module Documentation**: Added module-level docstrings to `glitchlab/display.py`, `glitchlab/controller_utils.py`, and `glitchlab/doc_inserter.py` to improve codebase maintainability and explain the role of each module in the controller decomposition.

## [4.5.0] - 2026-03-20

### Added
- **Failure Context Injection**: The controller now records failed task attempts to `.glitchlab/failures.jsonl`. If a new task objective matches a previous failure, the failure context (steps taken, files read/edited, reason) is automatically injected into the planner's prompt to prevent repeating mistakes.

## [4.4.0] - 2026-03-20

### Added
- **Per-Agent Token Tracking**: Added cumulative token usage tracking to `implementer` and `debugger` tool loops.
- **Loop Observability**: Introduced `agent.loop_step` events emitted after each LLM call, providing real-time step-by-step token consumption and cumulative totals.
- **Enhanced Metadata**: Agent results now include `_loop_tokens` in the return dictionary for final reporting.

## [4.3.0] - 2026-03-14

### Changed
- **Version Bump**: Updated version string to 4.3.0 across all relevant locations (pyproject.toml, glitchlab/__init__.py, glitchlab/identity.py, tests/test_identity.py).

## [4.2.0] - 2026-03-07

### Added
- **Zephyr SBOF Integration**: Introduced cryptographic signing and attestation for all agent actions. Every tool call, plan step, and code mutation is now signed with a tamper-evident signature before being committed to the event log. Provides cryptographic attestation, tamper detection, and audit-ready provenance for supply-chain security compliance.
- **EventBus Architecture Upgrade**: Enhanced the internal EventBus with three new first-class fields on every event:
  - `run_id` (UUID): Uniquely identifies a single end-to-end agent loop execution
  - `action_id` (UUID): Uniquely identifies each discrete action within a run
  - `metadata` (dict): Arbitrary structured context (model, token counts, timestamps, etc.)
  - These fields enable perfect deterministic traceability of agent loops, allowing reconstruction of exact action sequences, loop replay with identical inputs, and action-level diffing between runs.

## [4.1.0] - 2026-03-06

### Added
- **New Agents**: Added 'Shield' (TestGen role, Gemini model) and 'Archivist Nova' (Archivist role, Gemini model) to the Agent Roster.
- **Documentation**: Updated README.md to reflect v4.1.0 and expanded Agent Roster table with new agents.

## [4.0.0] - 2026-03-05

- **Auto-Merge Pipeline**: Added `auto_merge_pr` capability via `gh pr merge --squash` to enable fully hands-free PR merging.
- **Conflict Prevention**: Added `rebase_before_pr` to safely rebase worktrees against `origin/main` before opening PRs.
- **Self-Healing Batch Queue**: The `glitchlab batch` command now automatically catches rebase conflicts and requeues tasks to run against the newly updated `main` branch.
- **Task Priority Queue**: Batch tasks are now sorted by risk (High risk runs first when main is cleanest, Low risk runs last).
- **CLI Automation Flags**: Added `--auto-merge` to `run`, `batch`, and `interactive` commands.
- **API Resilience**: Added a strict 120-second timeout to the LiteLLM router to prevent silent API hangs.

## [3.1.1] - 2026-03-04

### Security
- Bump `litellm` to `1.82.0` to patch Critical CVE-2024-5751 (RCE) and CVE-2024-6587 (SSRF).
- Bump `GitPython` to `3.1.46` to patch High CVE-2024-22190 (Untrusted search path).
- Bump `loguru` to `0.7.3` to patch Medium CVE-2022-0338 (Sensitive info logging).
