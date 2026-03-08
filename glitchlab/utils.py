"""Utility helpers for GlitchLab."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable string.

    Examples::

        >>> format_duration(0)
        '0s'
        >>> format_duration(45)
        '45s'
        >>> format_duration(150.7)
        '2m 30s'
        >>> format_duration(3912)
        '1h 5m 12s'

    Args:
        seconds: Total duration in seconds (may be fractional; sub-second
                 precision is truncated).

    Returns:
        A string such as ``"45s"``, ``"2m 30s"``, or ``"1h 5m 12s"``.
    """
    total_seconds = int(seconds)

    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
