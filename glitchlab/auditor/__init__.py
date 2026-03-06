"""
GLITCHLAB Auditor — Scout Edition

Multi-layer autonomous codebase analysis engine.

Layer 1: Static analysis (tree-sitter + regex) — fast, no API calls
Layer 2: Dependency vulnerability scanning (OSV.dev) — no API key needed
Layer 3: LLM-powered creative analysis (Scout Brain) — requires router

Usage:
    glitchlab audit --repo ~/project                    # Layers 1+2 (default)
    glitchlab audit --repo ~/project --scout            # All 3 layers
    glitchlab audit --repo ~/project --kind test_gap    # Filter by finding kind
    glitchlab audit --repo ~/project --dry-run          # Preview only
"""

from .scanner import Scanner, ScanResult, Finding, FINDING_KINDS
from .task_writer import TaskWriter, group_findings_into_tasks, compute_priority

__all__ = [
    "Scanner",
    "ScanResult",
    "Finding",
    "FINDING_KINDS",
    "TaskWriter",
    "group_findings_into_tasks",
    "compute_priority",
]
