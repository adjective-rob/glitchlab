"""
Task Quality Gate — Catch vague or ambiguous task objectives.

Runs before a task enters the pipeline. Checks for language patterns
that cause implementer exploration spirals. Injects constraints when
ambiguity is detected.
"""

from __future__ import annotations

import re

from loguru import logger


# Patterns that correlate with implementer exploration spirals
_AMBIGUOUS_PATTERNS = [
    (r"\bclean\b", "clean"),
    (r"\bappropriate\b", "appropriate"),
    (r"\bsuitable\b", "suitable"),
    (r"\bproperly\b", "properly"),
    (r"\bcorrectly\b", "correctly"),
    (r"\bas needed\b", "as needed"),
    (r"\bnecessary changes\b", "necessary changes"),
    (r"\brefactor .* to be better\b", "refactor to be better"),
    (r"\bimprove the\b", "improve the"),
    (r"\boptimize the\b", "optimize the"),
    (r"\bupdate .* accordingly\b", "update accordingly"),
    (r"\bhandle .* gracefully\b", "handle gracefully"),
    (r"\bmake .* robust\b", "make robust"),
    (r"\bensure .* works\b", "ensure works"),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), label) for p, label in _AMBIGUOUS_PATTERNS]


def check_task_quality(objective: str) -> tuple[bool, list[str]]:
    """
    Check a task objective for ambiguous language.

    Returns:
        (is_clean, found_patterns) — is_clean is True if no ambiguity detected
    """
    found = []
    for pattern, label in _COMPILED_PATTERNS:
        if pattern.search(objective):
            found.append(label)
    return (len(found) == 0, found)


def get_quality_constraints(objective: str) -> list[str]:
    """
    If the objective is ambiguous, return constraints to inject into the task.
    Returns empty list if the objective is clean.
    """
    is_clean, patterns = check_task_quality(objective)
    if is_clean:
        return []

    logger.warning(
        f"[QUALITY] Task objective contains ambiguous language: {', '.join(patterns)}. "
        "Injecting narrow-interpretation constraints."
    )

    return [
        "This task objective contains ambiguous language. Interpret it NARROWLY.",
        "Only modify files explicitly named in the objective or plan. Do not explore.",
        "If the objective says 'clean up' or 'improve', limit changes to removing dead code, "
        "fixing lint errors, or adding missing type hints. Do not redesign.",
        "If you are unsure what the objective means, make the SMALLEST possible change "
        "that addresses the literal text. Do not guess at intent.",
        f"Ambiguous terms detected: {', '.join(patterns)}. Treat these as 'make minimal targeted edits'.",
    ]
