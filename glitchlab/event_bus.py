import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class GlitchEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    action_id: Optional[str] = None
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)  # For Zephyr signatures & scores


class EventBus:
    """A lightweight, synchronous event bus for decoupling GlitchLab observability."""
    
    def __init__(self):
        self._subscribers: List[Callable[[GlitchEvent], None]] = []

    def subscribe(self, callback: Callable[[GlitchEvent], None]) -> None:
        """Register a callback to be executed when an event is emitted."""
        self._subscribers.append(callback)

    def emit(
        self, 
        event_type: str, 
        payload: Dict[str, Any],
        agent_id: str = "system",
        run_id: str = None,
        action_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Construct and broadcast a GlitchEvent to all subscribers."""
        event = GlitchEvent(
            event_type=event_type,
            run_id=run_id,
            agent_id=agent_id,
            action_id=action_id,
            payload=payload,
            metadata=metadata or {}
        )
        
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception:
                # Prevent a failing subscriber (like a bad file write) from crashing the engine
                pass

# Global singleton instance for easy imports across the project
bus = EventBus()