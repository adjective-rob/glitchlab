import json
import os
from typing import Any, Dict

class AuditLogger:
    """
    Audit Logger that subscribes to an Event Bus and writes events
    to an append-only JSONL file.
    """
    def __init__(self, file_path: str, event_bus: Any):
        self.file_path = file_path
        self.event_bus = event_bus
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        
        # Subscribe to all events on the event bus
        # Assuming the event bus has a subscribe method that accepts a wildcard or similar
        self.event_bus.subscribe("*", self.log_event)

    def log_event(self, event: Dict[str, Any]) -> None:
        """
        Callback to handle incoming events and append them to the JSONL file.
        """
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + '\n')
