import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, Field


class GlitchEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str
    agent_name: str
    payload: Dict[str, Any]


class EventBus:
    """A lightweight, synchronous event bus for decoupling GlitchLab observability."""
    
    def __init__(self):
        self._subscribers: List[Callable[[GlitchEvent], None]] = []

    def subscribe(self, callback: Callable[[GlitchEvent], None]) -> None:
        """Register a callback to be executed when an event is emitted."""
        self._subscribers.append(callback)

    def emit(self, event_type: str, agent_name: str, payload: Dict[str, Any]) -> None:
        """Construct and broadcast a GlitchEvent to all subscribers."""
        event = GlitchEvent(
            event_type=event_type,
            agent_name=agent_name,
            payload=payload
        )
        
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception as e:
                # Prevent a failing subscriber (like a bad file write) from crashing the engine
                # In a real system we'd log this, but we keep it simple for now
                pass


# Global singleton instance for easy imports across the project
bus = EventBus()