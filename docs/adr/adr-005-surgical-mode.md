# ADR 005: Introduction of Surgical Mode in GlitchLab

Date: 2026-03-26

## Status

Accepted

## Context

GlitchLab needed a new mode to support a streamlined, minimal fix pipeline called "surgical mode." This mode is intended to apply focused fixes with minimal planning and limited fix attempts, improving efficiency for certain maintenance tasks.

## Decision

- Added an optional `profile` parameter to the `load_config()` function in `config_loader.py` to support loading and overlaying configuration profiles from `glitchlab/profiles/{profile}.yaml`.
- Introduced a new boolean `surgical` field in the `RunContext` dataclass to track surgical mode state.
- Extended the CLI commands `run` and `interactive` to accept a `--surgical` option, passing this flag to the `Controller`.
- Modified the `Controller` to accept a `surgical` parameter, store it, and alter its `run()` method behavior when surgical mode is enabled:
  - Load the `surgical` profile configuration.
  - Replace the pipeline in the run context with the surgical pipeline.
  - Limit fix attempts to 1.
  - Skip the planning step and proceed directly to pipeline execution.

This design leverages existing configuration and pipeline mechanisms, minimizing disruption to existing code and workflows.

## Consequences

- Surgical mode can be enabled via CLI or programmatically.
- Configuration profiles allow flexible overlays for different run modes.
- The controller's run logic is more complex but supports targeted execution paths.
- Existing pipelines and agents remain unchanged.

## Related

- Issue/Task: interactive-20260326-185939
- Files modified: `config_loader.py`, `run_context.py`, `cli.py`, `controller.py`
