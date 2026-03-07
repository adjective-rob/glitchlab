import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger
from glitchlab.event_bus import bus, GlitchEvent

class AuditLogger:
    """
    Acts as the Zephyr Attestation subscriber. 
    Listens to the EventBus, cryptographically signs tool actions using Zephyr, 
    and appends verified events to an audit trail.

    Gracefully degrades to standard JSONL logging if Zephyr is not installed.
    """
    def __init__(self, log_file="audit.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if Zephyr is available in the system PATH
        self.zephyr_enabled = shutil.which("zephyr") is not None
        
        if self.zephyr_enabled:
            logger.info("[AUDIT] Zephyr engine detected. Cryptographic SBOF signing enabled.")
        else:
            logger.warning("[AUDIT] Zephyr not found in PATH. Falling back to unsigned JSONL logging.")
        
        # Subscribe to the central EventBus
        bus.subscribe(self.log_event)

    def log_event(self, event: GlitchEvent) -> None:
        """
        Callback to handle incoming events, attempt Zephyr signing, 
        and append to the JSONL log.
        """
        # Serialize the GLITCHLAB event to use as the Zephyr payload
        event_json = event.model_dump_json()

        # --- ZEPHYR HARDWARE SIGNING ---
        if self.zephyr_enabled:
            # Use action_id for specific tools, otherwise use the parent event_id
            label = event.action_id if event.action_id else event.event_id
            
            try:
                # Synchronous Bridge: Pass the event to Zephyr to wrap in an Envelope and append
                subprocess.run(
                    [
                        "zephyr", "sign-payload",
                        "--label", label,
                        "--inline", event_json,
                        "--output", str(self.log_file),
                        "--append"
                    ],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return  # Zephyr successfully securely wrote to the file
            
            except subprocess.CalledProcessError as e:
                logger.error(f"[AUDIT] Zephyr signing failed: {e.stderr.strip()}")
                # Fallthrough to manual write to ensure we don't lose the log
            except Exception as e:
                logger.error(f"[AUDIT] Zephyr execution error: {e}")
                # Fallthrough to manual write

        # --- FALLBACK: MANUAL UNSIGNED WRITE ---
        # If Zephyr is missing, uninstalled, or failed, we write the raw event directly.
        event.metadata["zephyr_attestation"] = {
            "status": "unsigned_fallback",
            "reason": "Zephyr CLI unavailable"
        }
        
        # Write the fallback JSON with the updated metadata
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(event.model_dump_json() + '\n')