# Changelog

## [Unreleased]

### Added

- Introduced surgical mode to GlitchLab, enabling a focused minimal fix pipeline.
- Added `--surgical` CLI option to `run` and `interactive` commands.
- Added profile support in config loading to overlay surgical pipeline configuration.
- Controller run logic updated to support surgical mode execution path with limited fix attempts and no planning step.
- Added event bus publish/subscribe coverage test verifying subscriber callbacks receive emitted events in order.

### Changed

- RunContext now includes a `surgical` boolean flag to track surgical mode state.

### Version Bump

- Minor version bump due to new feature addition.
