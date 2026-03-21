# Task 16: Fix pre-existing lint warnings

**Files:** `glitchlab/controller.py`

**Problem:** There are 3 f-strings that contain no expressions (ruff F541). These are `console.print` calls using f-strings unnecessarily — the strings have no `{}` interpolation.

**What to change:**

Find and fix each one. In every case, just remove the `f` prefix from the string.

### Fix 1

Find (in `_finalize` method):
```python
                    console.print(f"[dim]🚀 Auto-merge enabled. Squashing and merging...[/]")
```
Replace with:
```python
                    console.print("[dim]🚀 Auto-merge enabled. Squashing and merging...[/]")
```

### Fix 2

Find (in `_finalize` method):
```python
                        console.print(f"[bold green]🎉 PR Auto-Merged successfully![/]")
```
Replace with:
```python
                        console.print("[bold green]🎉 PR Auto-Merged successfully![/]")
```

### Fix 3

Find (in `_run_fix_loop` or retry logic):
```python
                console.print(f"[bold blue]🔄 Resuming Patch...[/]")
```
Replace with:
```python
                console.print("[bold blue]🔄 Resuming Patch...[/]")
```

### Do NOT touch

- Any logic, just the unnecessary `f` prefixes on these 3 strings.

### Verify

```bash
python -c "from glitchlab.controller import Controller; print('ok')"
# Check no more expressionless f-strings (these 3 specific lines)
python3 -c "
import ast
with open('glitchlab/controller.py') as f:
    tree = ast.parse(f.read())
count = 0
for node in ast.walk(tree):
    if isinstance(node, ast.JoinedStr) and all(isinstance(v, ast.Constant) for v in node.values):
        count += 1
        print(f'  F541 at line {node.lineno}')
print(f'Expressionless f-strings remaining: {count}')
"
# Expected: 0
python -m pytest tests/ -x
```
