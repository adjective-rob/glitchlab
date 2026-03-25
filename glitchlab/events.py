"""
GLITCHLAB Event Emitter — Unified event logging.

Every meaningful event in the pipeline flows through emit_event(), which:
  1. Appends to TaskState.events (structured in-memory log)
  2. Broadcasts via EventBus (Zephyr signs it, audit logger persists it)

This replaces the three separate _log_event helpers that only did #1.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from loguru import logger

from glitchlab.event_bus import bus

if TYPE_CHECKING:
    from glitchlab.run_context import RunContext


def emit_event(
    ctx: "RunContext",
    event_type: str,
    data: dict[str, Any] | None = None,
    *,
    agent_id: str = "controller",
) -> None:
    """Log an event to TaskState AND broadcast it via the EventBus.

    This is the single chokepoint for all pipeline observability.
    Every call here produces a Zephyr-signable event.
    """
    now = datetime.now(timezone.utc).isoformat()
    payload = data or {}

    # 1. Append to TaskState (in-memory structured log)
    event_record = {
        "type": event_type,
        "timestamp": now,
        "task_id": ctx.state.task_id if ctx.state else None,
        "data": payload,
    }
    if ctx.state:
        ctx.state.events.append(event_record)

    # 2. Broadcast via EventBus (Zephyr signs, audit logger persists)
    bus.emit(
        event_type=event_type,
        payload=payload,
        agent_id=agent_id,
        run_id=ctx.run_id,
    )

    logger.debug(f"[EVENT] {event_type}: {payload}")