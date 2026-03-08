# ADR-0003: Extended GlitchEvent Schema with Traceability Fields

**Date:** 2025-03-08  
**Status:** Accepted  
**Context:** Task ID interactive-20260308-005714  

## Problem

The original `GlitchEvent` data model lacked sufficient traceability and correlation information needed for:
- Linking events to specific execution runs
- Correlating events with specific actions within a run
- Attaching arbitrary metadata for extensibility and debugging

## Decision

Extended the `GlitchEvent` class and `EventBus.emit()` method signature to include three new fields:

1. **`run_id` (str, required)**: Unique identifier for the execution run that generated the event. Enables grouping all events from a single run.

2. **`action_id` (Optional[str])**: Unique identifier for the specific action/step within a run that generated the event. Enables fine-grained event correlation.

3. **`metadata` (dict)**: Arbitrary key-value pairs for extensible event context. Allows attaching source information, custom tags, or debugging data without schema changes.

Additionally, the field `agent_name` was renamed to `agent_id` for consistency with the new naming convention.

## Rationale

- **Traceability**: `run_id` enables tracking all events from a single execution, critical for debugging and audit trails.
- **Correlation**: `action_id` allows linking events to specific steps, useful for understanding event sequences.
- **Extensibility**: `metadata` dict avoids future schema changes by allowing arbitrary context attachment.
- **Consistency**: Renaming `agent_name` to `agent_id` aligns with the new field naming pattern.

## Consequences

### Positive
- Events are now fully traceable to their originating run and action
- System is more extensible without requiring schema changes
- Better support for multi-run and multi-action scenarios

### Negative
- All code that instantiates `GlitchEvent` or calls `EventBus.emit()` must be updated
- Existing event logs may not have these fields (backward compatibility concern)

## Migration Path

All test files and event emission sites have been updated to include the new fields:
- `run_id`: Set to a meaningful run identifier (e.g., "run-001")
- `action_id`: Set to action identifier or None if not applicable
- `metadata`: Set to a dict with relevant context (e.g., `{"source": "test"}`)

## References

- Implementation: `glitchlab/event_bus.py` - GlitchEvent class and EventBus.emit() method
- Tests: `tests/test_event_bus.py` - Updated test cases demonstrating usage
