# ADR: Synchronize version reporting across package metadata and public exports

Date: 2026-03-31
Task ID: interactive-20260331-151538

## Status
Accepted

## Context
The project exposes version information in multiple places:
- `pyproject.toml` for package metadata
- `glitchlab/__init__.py` for public package export
- `glitchlab/identity.py` for identity/version checks
- `tests/test_identity.py` for regression protection

During the upgrade from 4.4.0 to 4.5.0, these values needed to remain aligned to avoid version drift and false failures in identity checks.

## Decision
Keep the canonical version synchronized across all public version surfaces whenever the project version changes:
- package metadata
- top-level package export
- identity module export
- identity test expectations

## Consequences
- Version upgrades must update all listed files together.
- Tests continue to enforce consistency between the runtime version and the expected release version.
- Reduces the chance of shipping mismatched version strings across modules.

## Related files
- `pyproject.toml`
- `glitchlab/__init__.py`
- `glitchlab/identity.py`
- `tests/test_identity.py`
