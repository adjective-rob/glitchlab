"""Tests for parallel phase execution and thread-safe budget tracking."""

import threading
import time
from dataclasses import dataclass, field
from unittest.mock import MagicMock

from glitchlab.router import BudgetTracker, UsageRecord


def test_budget_tracker_thread_safe():
    """Multiple threads recording usage concurrently should not lose data."""
    tracker = BudgetTracker(max_tokens=1_000_000, max_dollars=100.0)

    num_threads = 10
    records_per_thread = 50

    def make_mock_response(tokens: int):
        mock = MagicMock()
        mock.usage.prompt_tokens = tokens
        mock.usage.completion_tokens = tokens
        mock.usage.total_tokens = tokens * 2
        return mock

    def record_many(role: str):
        for _ in range(records_per_thread):
            resp = make_mock_response(10)
            tracker.record(resp, role)

    threads = [
        threading.Thread(target=record_many, args=(f"agent-{i}",))
        for i in range(num_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each thread records 50 times, each with 20 total_tokens
    expected_total_tokens = num_threads * records_per_thread * 20
    expected_call_count = num_threads * records_per_thread

    summary = tracker.summary()
    assert summary["total_tokens"] == expected_total_tokens
    assert summary["call_count"] == expected_call_count


def test_budget_tracker_summary_under_contention():
    """summary() should not raise even when called concurrently with record()."""
    tracker = BudgetTracker()
    errors = []

    def record_loop():
        for _ in range(100):
            mock = MagicMock()
            mock.usage.prompt_tokens = 5
            mock.usage.completion_tokens = 5
            mock.usage.total_tokens = 10
            try:
                tracker.record(mock, "agent")
            except Exception as e:
                errors.append(e)

    def summary_loop():
        for _ in range(100):
            try:
                tracker.summary()
            except Exception as e:
                errors.append(e)

    t1 = threading.Thread(target=record_loop)
    t2 = threading.Thread(target=summary_loop)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0


def test_budget_exceeded_is_correct():
    """Verify budget_exceeded flag works correctly with lock."""
    tracker = BudgetTracker(max_tokens=100, max_dollars=100.0)

    mock = MagicMock()
    mock.usage.prompt_tokens = 60
    mock.usage.completion_tokens = 60
    mock.usage.total_tokens = 120
    tracker.record(mock, "agent")

    assert tracker.budget_exceeded is True
    assert tracker.tokens_remaining == 0


def test_parallel_phases_independence():
    """
    Simulate the parallel Security + Test pattern:
    two functions running concurrently should both complete
    and their results should be independently collected.
    """
    results = {}

    def fake_security():
        time.sleep(0.05)  # simulate LLM call
        results["security"] = {"verdict": "pass", "issues": []}

    def fake_test_loop():
        time.sleep(0.03)  # simulate test run
        results["test"] = {"passed": True, "attempts": 1}

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        sec_future = executor.submit(fake_security)
        test_future = executor.submit(fake_test_loop)

        # Both should complete without error
        sec_future.result(timeout=5)
        test_future.result(timeout=5)

    assert results["security"]["verdict"] == "pass"
    assert results["test"]["passed"] is True


def test_parallel_with_one_failure():
    """If one parallel phase fails, the other should still complete."""

    def failing_security():
        raise RuntimeError("Model API timeout")

    def succeeding_test():
        return {"passed": True}

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        sec_future = executor.submit(failing_security)
        test_future = executor.submit(succeeding_test)

        test_result = test_future.result(timeout=5)
        assert test_result["passed"] is True

        # Security should raise
        try:
            sec_future.result(timeout=5)
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "timeout" in str(e)
