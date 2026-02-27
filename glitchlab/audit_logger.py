import os
import time

class AuditLogger:
    def __init__(self, log_file="audit.log", batch_size=10):
        self.log_file = log_file
        self.batch_size = batch_size
        self._buffer = []

    def log(self, message):
        self._buffer.append(f"[{time.time()}] {message}\n")
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        with open(self.log_file, "a") as f:
            f.writelines(self._buffer)
        self._buffer.clear()

    def __del__(self):
        self.flush()
