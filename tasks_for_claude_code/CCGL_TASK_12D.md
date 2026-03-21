## Task 12d: Extract `_finalize()` from `Controller.run()`

**File:** `glitchlab/controller.py`

**What:** Extract the doc-only defaults, commit, rebase, PR creation, budget summary, and task archival into `_finalize()`. Pure refactor.

### Step 1: Add the new method

Insert directly above `_startup`. The method signature:

```python
    def _finalize(self, task: Task, plan: dict, impl: dict, rel: dict, sec: dict,
                  ws_path: Path, is_doc_only: bool, is_fast_mode: bool, result: dict) -> dict:
        """Commit changes, create PR, archive task. Returns updated result dict."""
```

The method body is the exact code currently in `run()` starting from:

```python
            # ── Phase routing: doc-only defaults for downstream ──
            if is_doc_only:
```

...through and including:

```python
                console.print(f"[dim]Moved task file to {archive_dir / task.file_path.name}[/]")
```

(The line just before `except BudgetExceededError`.)

Move all of that code into `_finalize`, **dedented by one level** (from 3x indent inside `try:` to 2x indent inside the method body). Add `return result` at the end.

### Step 2: Replace inline block in `run()`

Replace the entire block you just extracted (from `# ── Phase routing:` through the task archive print) with:

```python
            result = self._finalize(task, plan, impl, rel, sec, ws_path, is_doc_only, is_fast_mode, result)
```

### Important detail

The extracted block contains `if pipeline_halted: return result`. Inside `_finalize`, this needs to stay as `return result` — it already returns the result dict, and the caller in `run()` assigns `result = self._finalize(...)`, so the early return works correctly.

### Do NOT touch

- The pipeline loop, `_startup`, `_check_repo_clean`, `_print_banner`, the `except`/`finally` blocks in `run()`.

### Verify

```bash
python -c "from glitchlab.controller import Controller; print('ok')"
grep -n "def _finalize" glitchlab/controller.py          # exists
grep -c "Phase routing" glitchlab/controller.py          # expect 1 (in new method)
grep -c "Commit + PR" glitchlab/controller.py            # expect 1 (in new method)
grep -c "Auto-Merge" glitchlab/controller.py             # expect 1 (in new method)
python -m pytest tests/ -x
```