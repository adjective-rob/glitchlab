
## Task 12e: Extract `_execute_pipeline()` from `Controller.run()`

**File:** `glitchlab/controller.py`

**What:** Extract the dynamic pipeline loop and all per-agent post-processing into `_execute_pipeline()`. This is the last extraction — after this, `run()` becomes a short orchestrator. Pure refactor.

### Step 1: Add the new method

Insert directly above `_finalize`. Signature:

```python
    def _execute_pipeline(self, task: Task, ws_path: Path, tools: "ToolExecutor",
                          failure_context: str, result: dict) -> tuple[dict, dict, dict, dict, list, bool, bool, bool, dict]:
        """Run the dynamic pipeline. Returns (plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, result)."""
```

The body is the exact code currently in `run()` starting from:

```python
            # ── 2. Dynamic Pipeline ──
            plan: dict = {}
```

...through the end of the `for step in self.config.pipeline:` loop and all per-agent post-processing that follows it, ending just before the `# ── Phase routing:` comment (which is now in `_finalize`).

Dedent by one level. Add at the end:

```python
        return plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, result
```

### Step 2: Replace inline block in `run()`

Replace everything from `# ── 2. Dynamic Pipeline ──` through the line just before the `_finalize` call with:

```python
            plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, result = self._execute_pipeline(
                task, ws_path, tools, failure_context, result
            )
```

### What `run()` should now look like inside `try:`

```python
        try:
            ws_path, tools, failure_context = self._startup(task)

            plan, impl, rel, sec, applied, is_doc_only, is_fast_mode, test_ok, result = self._execute_pipeline(
                task, ws_path, tools, failure_context, result
            )

            result = self._finalize(task, plan, impl, rel, sec, ws_path, is_doc_only, is_fast_mode, result)
```

### Do NOT touch

- `_startup`, `_finalize`, `_check_repo_clean`, `_print_banner`, `_run_pipeline_step`, `_run_planner`, `_run_implementer`, or any other existing method.

### Verify

```bash
python -c "from glitchlab.controller import Controller; print('ok')"
grep -n "def _execute_pipeline" glitchlab/controller.py  # exists
grep -c "Dynamic Pipeline" glitchlab/controller.py       # expect 1 (in new method)

# run() should now be short
awk '/def run\(self, task/,/^    def [^_]/' glitchlab/controller.py | wc -l
# Expected: ~60-80 lines (the orchestrator)

# Method count increased
grep -c "def " glitchlab/controller.py
# Expected: ~5 more than original

python -m pytest tests/ -x
```