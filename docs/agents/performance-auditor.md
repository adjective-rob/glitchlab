# Performance Auditor Agent

## Overview
The Performance Auditor Agent is designed to identify I/O inefficiencies and potential resource leaks within the codebase.

## Capabilities
- **I/O Analysis**: Detects redundant read/write operations.
- **Leak Detection**: Identifies unclosed streams, sockets, and file handles.
- **Model**: Powered by `gemini-3.5-flash` for high-speed inference.

## Usage
```python
from glitchlab.agents import PerformanceAuditorAgent

auditor = PerformanceAuditorAgent()
report = auditor.audit(code_snippet)
```