# Task 15: Add hard write-deadline circuit breaker to implementer

**File:** `glitchlab/agents/implementer.py`

**Problem:** The implementer has a soft stall nudge at step 15 that appends a warning message, but the agent can ignore it and keep reading until it hits `max_steps` (30). This burns budget on loops where the agent never writes. We need a hard cutoff: if the agent reaches step 20 with `write_count == 0`, terminate the loop immediately with a failure result.

**What to change:**

Find this block inside `run()` (the existing stall detection):

```python
            # Stall detection: if we're past step 15 with no writes, force a think+act nudge
            if step >= 15 and write_count == 0 and think_count > 0:
                messages.append({
                    "role": "user",
                    "content": (
                        "WARNING: You have spent 15+ steps reading and searching without writing any code. "
                        "You MUST take action now. Use `replace_in_file` to make your changes, or call `done` "
                        "if the task cannot be completed. Do NOT read more files."
                    )
                })
```

Replace with:

```python
            # Stall detection: soft nudge at 15, hard kill at 20
            if step >= 15 and write_count == 0 and think_count > 0:
                if step >= 20:
                    # Hard circuit breaker — agent is stuck in a read loop
                    logger.warning(
                        f"[PATCH] Write-deadline breaker tripped at step {step + 1}. "
                        f"0 writes after {think_count} thinks and {search_count} searches."
                    )
                    bus.emit(
                        event_type="agent.write_deadline_breaker",
                        payload={
                            "step": step + 1,
                            "think_count": think_count,
                            "search_count": search_count,
                            "write_count": 0,
                        },
                        agent_id=self.role,
                    )
                    return {
                        "changes": [],
                        "tests_added": [],
                        "commit_message": "chore: no changes (write deadline exceeded)",
                        "summary": f"Implementer read for {step + 1} steps without writing. Circuit breaker tripped.",
                        "parse_error": True,
                        "_breaker": "write_deadline",
                    }
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            "WARNING: You have spent 15+ steps reading and searching without writing any code. "
                            "You MUST take action now. Use `replace_in_file` to make your changes, or call `done` "
                            "if the task cannot be completed. Do NOT read more files."
                        )
                    })
```

**What changed and why:**

- The existing soft nudge (step 15) is preserved but moved into the `else` branch.
- A new hard breaker fires at step 20 with `write_count == 0`. It logs, emits an event for observability, and returns a failure result.
- The failure result includes `"_breaker": "write_deadline"` so the controller/history can distinguish this from a normal max-step exhaustion.

### Do NOT touch

- The `done` handler, the max-step exhaustion fallback at the end of the loop, or any other part of `run()`.
- Any other file.

### Verify

```bash
python -c "from glitchlab.agents.implementer import ImplementerAgent; print('ok')"
grep -n "write_deadline" glitchlab/agents/implementer.py  # should show 2-3 lines (breaker code)
grep -c "step >= 20" glitchlab/agents/implementer.py      # expect 1
python -m pytest tests/ -x
```
