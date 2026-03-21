
## Task 12c: Extract `_startup()` from `Controller.run()`

**File:** `glitchlab/controller.py`

**What:** Extract workspace creation, repo indexing, scope resolver init, Prelude constraint loading, and failure context loading into `_startup()`. Returns `(ws_path, tools, failure_context)`. Pure refactor.

### Step 1: Add the new method

Insert it directly above `run()`. Find anchor:

```
    def run(self, task: Task) -> dict[str, Any]:
        """Execute the full agent pipeline for a task."""
```

Insert above it:

```python
    def _startup(self, task: Task) -> tuple[Path, "ToolExecutor", str]:
        """Create workspace, build indexes, load constraints. Returns (ws_path, tools, failure_context)."""
        # ── 1. Create workspace ──
        self._workspace = Workspace(
            self.repo_path, task.task_id,
            self.config.workspace.worktree_dir,
        )
        ws_path = self._workspace.create()
        self._log_event("workspace_created", {"path": str(ws_path)})

        tools = ToolExecutor(
            allowed_tools=self.config.allowed_tools,
            blocked_patterns=self.config.blocked_patterns,
            working_dir=ws_path,
        )

        # ── 1.5. Build repo index (file map for planner) ──
        console.print("\n[bold dim]🗂  [INDEX] Scanning repository...[/]")
        self._repo_index = build_index(ws_path)
        self._repo_index_context = self._repo_index.to_agent_context(max_files=200)
        console.print(
            f"  [dim]{self._repo_index.total_files} files, "
            f"{len(self._repo_index.languages)} languages[/]"
        )
        self._log_event("repo_indexed", {
            "total_files": self._repo_index.total_files,
            "languages": self._repo_index.languages,
        })

        # ── 1.6. Initialize ScopeResolver (Layer 1) ──
        self._scope = ScopeResolver(ws_path, self._repo_index)

        # ── 1.7. Prelude: load constraints only (not global prefix) ──
        if self._prelude.available:
            console.print("[bold dim]📋 [PRELUDE] Loading constraints...[/]")
            self._prelude.refresh()
            prelude_constraints = self._prelude.get_constraints()
            if prelude_constraints:
                task.constraints = list(set(task.constraints + prelude_constraints))
                console.print(f"  [dim]{len(prelude_constraints)} constraints merged[/]")
            self._log_event("prelude_constraints_loaded", {
                "count": len(prelude_constraints) if prelude_constraints else 0,
            })

        # ── 1.8. Load failure context from history ──
        failure_context = self._history.build_failure_context()
        if failure_context:
            console.print("  [dim]Loaded recent failure patterns for planner[/]")

        return ws_path, tools, failure_context

```

### Step 2: Replace inline block in `run()`

Find the block that starts with `# ── 1. Create workspace ──` and ends just before `# ── 2. Dynamic Pipeline ──`. The exact anchor lines:

**Start (find this):**
```python
            # ── 1. Create workspace ──
            self._workspace = Workspace(
```

**End (find the line before this):**
```python
            # ── 2. Dynamic Pipeline ──
            plan: dict = {}
```

Replace everything from `# ── 1. Create workspace ──` through the blank line after `console.print("  [dim]Loaded recent failure patterns for planner[/]")` with:

```python
            ws_path, tools, failure_context = self._startup(task)
```

So the `try:` block now reads:
```python
        try:
            ws_path, tools, failure_context = self._startup(task)

            # ── 2. Dynamic Pipeline ──
            plan: dict = {}
```

### Do NOT touch

- `_check_repo_clean`, `_print_banner`, the pipeline loop, the finalize block, any other method.

### Verify

```bash
python -c "from glitchlab.controller import Controller; print('ok')"
grep -n "def _startup" glitchlab/controller.py       # exists
grep -c "Create workspace" glitchlab/controller.py   # expect 1 (in new method only)
grep -c "ScopeResolver" glitchlab/controller.py      # expect 2 (import + new method)
python -m pytest tests/ -x
```
