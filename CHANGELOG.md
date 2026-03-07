# Changelog

## [Unreleased]

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
