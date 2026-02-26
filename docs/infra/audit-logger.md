# Audit Logger

Task: `infra-013-audit-logger`  
Mode: evolution  
Risk: low  
Version bump: minor

## Overview
The `AuditLogger` subscribes to the Event Bus and persists every received event to an append-only JSON Lines (JSONL) file. Each event is written as a single line of JSON to support efficient streaming, grepping, and incremental ingestion.

## Behavior
- Subscribes to the `EventBus`.
- On each event:
  - Appends a JSON-serialized record to the configured `.jsonl` file.
  - File is treated as append-only (no mutation of existing lines).

## Output format
- One JSON object per line (JSONL).
- Payload corresponds to the event object as emitted on the bus (fields depend on the event schema used by the system).

## Configuration / Usage
Instantiate `AuditLogger` with:
- an `EventBus` instance to subscribe to
- a target file path for the JSONL output

(Refer to `audit_logger.py` for constructor details and integration points.)

## Operational notes
- The JSONL file can grow indefinitely; apply external rotation/retention policies as needed.
- Consumers should assume the log is append-only and process line-by-line.
