## Task 13c: Wire `symbol_index` into `_run_planner` context

**File:** `glitchlab/controller.py`

**What:** The planner's new tool loop needs `symbol_index` in its `AgentContext.extra` dict. Currently `SymbolIndex` is only instantiated inside `_run_implementer`. Add it to `_run_planner` too.

### Step 1: Add SymbolIndex instantiation to `_run_planner`

Find this block inside `_run_planner`:

```python
        context = AgentContext(
            task_id=task.task_id,
            run_id=self.run_id,
            objective=objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            constraints=task.constraints,
            acceptance_criteria=task.acceptance_criteria,
            risk_level=task.risk_level,
            extra={
                "prelude": self._prelude,
            },
        )
```

Replace with:

```python
        symbol_index = SymbolIndex(ws_path)

        context = AgentContext(
            task_id=task.task_id,
            run_id=self.run_id,
            objective=objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            constraints=task.constraints,
            acceptance_criteria=task.acceptance_criteria,
            risk_level=task.risk_level,
            extra={
                "prelude": self._prelude,
                "symbol_index": symbol_index,
            },
        )
```

Two changes: added `symbol_index = SymbolIndex(ws_path)` line before the context, and added `"symbol_index": symbol_index,` to the `extra` dict.

`SymbolIndex` is already imported at line 65: `from glitchlab.symbols import SymbolIndex`.

### Do NOT touch

- `_run_implementer` (it has its own `SymbolIndex` instantiation — leave it).
- The `raw = self.agents["planner"].run(context)` call or anything after it.

### Verify

```bash
python -c "from glitchlab.controller import Controller; print('ok')"
grep -n "symbol_index" glitchlab/controller.py
# Expected: 4 lines — import, _run_planner instantiation, _run_planner extra, _run_implementer instantiation, _run_implementer extra
grep -c "SymbolIndex(ws_path)" glitchlab/controller.py
# Expected: 2 (one in _run_planner, one in _run_implementer)
python -m pytest tests/ -x
```