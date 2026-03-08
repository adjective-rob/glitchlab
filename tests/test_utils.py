import pytest

from glitchlab.utils import format_duration


# ---------------------------------------------------------------------------
# Seconds-only
# ---------------------------------------------------------------------------

def test_zero_seconds():
    assert format_duration(0) == "0s"


def test_seconds_only():
    assert format_duration(45) == "45s"


def test_one_second():
    assert format_duration(1) == "1s"


def test_fifty_nine_seconds():
    assert format_duration(59) == "59s"


# ---------------------------------------------------------------------------
# Minutes + seconds
# ---------------------------------------------------------------------------

def test_exactly_one_minute():
    assert format_duration(60) == "1m 0s"


def test_minutes_and_seconds():
    assert format_duration(150) == "2m 30s"


def test_minutes_and_seconds_large():
    assert format_duration(3599) == "59m 59s"


# ---------------------------------------------------------------------------
# Hours + minutes + seconds
# ---------------------------------------------------------------------------

def test_exactly_one_hour():
    assert format_duration(3600) == "1h 0m 0s"


def test_hours_minutes_seconds():
    assert format_duration(3912) == "1h 5m 12s"


def test_hours_zero_minutes():
    assert format_duration(7205) == "2h 0m 5s"


# ---------------------------------------------------------------------------
# Fractional seconds (sub-second precision is truncated)
# ---------------------------------------------------------------------------

def test_fractional_seconds_truncated():
    # 150.7 seconds → same as 150 seconds → "2m 30s"
    assert format_duration(150.7) == "2m 30s"


def test_fractional_below_one_second():
    # 0.9 seconds → truncated to 0 → "0s"
    assert format_duration(0.9) == "0s"


def test_fractional_just_under_minute():
    # 59.99 → truncated to 59 → "59s"
    assert format_duration(59.99) == "59s"
