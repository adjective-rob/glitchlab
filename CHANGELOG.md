# Changelog

## [Unreleased]

## [4.3.0] - 2026-03-08

### Changed
- **YAML Agent Messaging**: Messages are now communicated across agents in YAML, providing a structured, human-readable interchange format for all inter-agent communication.
### Added
- **Duration Formatting Utility**: New `format_duration(seconds: float) -> str` function in `glitchlab/utils.py` converts seconds into human-readable duration strings (e.g., "2m 30s", "1h 5m 12s"). Includes comprehensive test coverage with 13 test cases.

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
