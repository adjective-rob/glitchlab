## Task 13b: Rewrite `PlannerAgent.run()` to use agentic tool loop

**File:** `glitchlab/agents/planner.py`

**What:** Replace the 4-line single-shot `run()` with an agentic loop that uses `PLANNER_TOOLS` to explore the codebase before submitting a plan. Also add two rules to the system prompt.

### Step 1: Add rules 15-16 to the system prompt

Find the end of the system prompt. The last rule currently is:

```
14. If the task mentions constraints about what NOT to change, those MUST appear in do_not_touch.
"""
```

Replace with:

```
14. If the task mentions constraints about what NOT to change, those MUST appear in do_not_touch.
15. You have tools to explore the codebase before submitting your plan. Use get_function to read the exact functions you plan to modify. Use get_class to understand class structures. Use search_grep to find where things are defined. Your code_hints MUST reference exact function names and line numbers from your exploration — not guesses.
16. When you are confident in your plan, call submit_plan with the complete JSON. Do not submit a plan without first reading the target functions.
"""
```

### Step 2: Replace `run()`

Find the existing `run()`:

```python
    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Override run to enforce JSON mode at the API level."""
        kwargs["response_format"] = {"type": "json_object"}
        return super().run(context, **kwargs)
```

Replace with:

```python
    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        """Execute the planner in an agentic loop with read-only tools."""
        import subprocess
        from pathlib import Path

        messages = self.build_messages(context)
        workspace_dir = Path(context.working_dir)
        symbol_index = context.extra.get("symbol_index")

        max_steps = 8

        for step in range(max_steps):
            step_kwargs = dict(kwargs)
            # Remove response_format — can't use JSON mode with tools
            step_kwargs.pop("response_format", None)

            if step == 0:
                step_kwargs["tool_choice"] = {"type": "function", "function": {"name": "think"}}

            response = self.router.complete(
                role=self.role,
                messages=messages,
                tools=PLANNER_TOOLS,
                **step_kwargs
            )

            # Append assistant message
            assist_msg: dict[str, Any] = {"role": "assistant"}
            if response.content:
                assist_msg["content"] = response.content
            if response.tool_calls:
                assist_msg["tool_calls"] = [
                    tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
                    for tc in response.tool_calls
                ]
            messages.append(assist_msg)

            if not response.tool_calls:
                # If the model returns plain text, try to parse it as JSON (fallback)
                if response.content:
                    return self.parse_response(response, context)
                messages.append({
                    "role": "user",
                    "content": "Use your tools to explore the codebase, then call submit_plan with your JSON plan.",
                })
                continue

            for tool_call in response.tool_calls:
                tc_id = tool_call.id
                tc_name = tool_call.function.name
                try:
                    tc_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": "Invalid JSON."})
                    continue

                if tc_name == "think":
                    messages.append({
                        "role": "tool", "tool_call_id": tc_id, "name": tc_name,
                        "content": "Analysis noted. Explore the codebase or submit your plan.",
                    })

                elif tc_name == "get_function":
                    symbol = tc_args.get("symbol")
                    file_path = tc_args.get("file")
                    if symbol_index:
                        func_data = symbol_index.get_function_body(symbol, file_path)
                        if func_data:
                            res = (
                                f"Function '{symbol}' in {func_data['file']} "
                                f"(Lines {func_data['line_start']}-{func_data['line_end']}):\n\n"
                                f"{func_data['body']}"
                            )
                        else:
                            res = f"Function '{symbol}' not found."
                    else:
                        res = "AST parser unavailable."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "get_class":
                    class_name = tc_args.get("class_name")
                    file_path = tc_args.get("file")
                    if symbol_index:
                        class_data = symbol_index.get_class_outline(class_name, file_path)
                        if class_data:
                            res = (
                                f"Class '{class_name}' in {class_data['file']} "
                                f"(Lines {class_data['line_start']}-{class_data['line_end']}):\n\n"
                                f"{class_data['outline']}"
                            )
                        else:
                            res = f"Class '{class_name}' not found."
                    else:
                        res = "AST parser unavailable."
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "search_grep":
                    pattern = tc_args.get("pattern")
                    file_type = tc_args.get("file_type", "*.py")
                    try:
                        cmd = [
                            "grep", "-rn", f"--include={file_type}",
                            "--exclude-dir=.glitchlab", "--exclude-dir=__pycache__",
                            "--exclude-dir=.git", pattern, ".",
                        ]
                        proc = subprocess.run(cmd, cwd=workspace_dir, capture_output=True, text=True, timeout=10)
                        lines = proc.stdout.splitlines()
                        res = "\n".join(lines[:30]) if lines else "No matches."
                    except Exception as e:
                        res = f"Search failed: {e}"
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res})

                elif tc_name == "submit_plan":
                    plan_json = tc_args.get("plan_json", "")
                    fake_response = RouterResponse(
                        content=plan_json,
                        model=response.model,
                        tokens_used=response.tokens_used,
                        cost=response.cost,
                        latency_ms=response.latency_ms,
                    )
                    return self.parse_response(fake_response, context)

        # Fallback: loop exhausted without submit_plan
        logger.warning("[ZAP] Planner loop exhausted without calling submit_plan.")
        return {
            "steps": [],
            "files_likely_affected": [],
            "requires_core_change": False,
            "risk_level": "high",
            "risk_notes": "Planner failed to produce a plan within step limit",
            "test_strategy": [],
            "estimated_complexity": "unknown",
            "parse_error": True,
        }
```

### Do NOT touch

- `build_messages`, `parse_response`, `_warn_on_quality_gaps`, `PlanStep`, `ExecutionPlan`.

### Verify

```bash
python -c "from glitchlab.agents.planner import PlannerAgent; print('ok')"
grep -c "submit_plan" glitchlab/agents/planner.py      # expect 3+ (tool def, handler, log)
grep -c "PLANNER_TOOLS" glitchlab/agents/planner.py    # expect 2 (definition, usage in run)
grep "rule 15\|rule 16\|get_function to read" glitchlab/agents/planner.py  # prompt rules exist
python -m pytest tests/ -x
```

---

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