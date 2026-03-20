# GLITCHLAB Agent Guide

This file is injected into agent prompts to reduce exploration steps. If you're an agent reading this, use it instead of reading entire files.

## File Map

| File | Purpose | Key functions |
|------|---------|---------------|
| `glitchlab/controller.py` | Pipeline orchestration. Runs all agents in sequence. | `run()` (main entry), `_run_pipeline_step()` (dispatcher), `_run_implementer()` (builds implementer context) |
| `glitchlab/agents/implementer.py` | Edit engine. Reads, writes, verifies code. | `build_messages()` (prompt construction), `run()` (tool loop), `IMPLEMENTER_TOOLS` (tool definitions list) |
| `glitchlab/agents/planner.py` | Planning. Produces JSON execution plans. | `build_messages()` (prompt construction), `parse_response()` (JSON validation), `PlanStep`/`ExecutionPlan` (Pydantic schemas) |
| `glitchlab/agents/debugger.py` | Fix engine. Diagnoses and fixes test failures. | `build_messages()`, `run()` (tool loop), `DEBUGGER_TOOLS` (tool definitions list) |
| `glitchlab/router.py` | LLM abstraction via LiteLLM. Budget tracking, retries. | `complete()` (send request), `ContextMonitor.enforce_headroom()`, `BudgetTracker` |
| `glitchlab/context_compressor.py` | Memory management. Shrinks old messages. | `compress_stale_messages()` (content compression), `hard_compact_messages()` (message removal) |
| `glitchlab/prelude.py` | Project context bridge to Prelude CLI. | `compact()`, `query()`, `get_constraints()`, `refresh()` |
| `glitchlab/symbols.py` | AST layer for code navigation via tree-sitter. | `get_function_body()`, `get_class_outline()`, `find_references()`, `invalidate()` |
| `glitchlab/event_bus.py` | Global event system. | `bus.emit(event_type, payload, agent_id)` |
| `glitchlab/runners.py` | Agent runner helpers called by controller. | `run_security()`, `run_release()`, `run_archivist()`, `run_delegated_agent()` |
| `glitchlab/config_loader.py` | YAML config parsing. | `GlitchLabConfig` (Pydantic model for glitchlab.yaml) |
| `glitchlab/workspace/tools.py` | Sandboxed command executor for `run_check`. | `ToolExecutor.execute()` |

## Controller Anatomy

`controller.py` is ~1300 lines. Here's how it flows:

1. **Startup** (lines ~120-240): Git checks, repo indexing, Prelude refresh, constraint loading, failure history loading
2. **Planning** (lines ~250-280): Builds planner AgentContext, calls planner agent, gets ExecutionPlan back
3. **Implementation** (lines ~280-330): Calls `_run_implementer()` which builds AgentContext with tool_executor, symbol_index, prelude, and file_context in the `extra` dict
4. **Testing** (lines ~330-380): Runs `python -m pytest`, retries up to 4 times, sends to debugger on failure
5. **Post-pipeline** (lines ~380-450): Testgen (regression tests), security scan, release assessment, archivist (docs/ADRs)
6. **Finalize** (lines ~450-500): Status assignment, PR creation, workspace cleanup

Key variables that carry state through the pipeline:
- `plan` — dict from planner, contains `steps`, `files_likely_affected`, `risk_level`
- `impl` — dict from implementer, contains `changes`, `commit_message`, `_loop_tokens`
- `self._state` — TaskState object, accumulates files_modified, test results, etc.
- `self._prelude` — PreludeContext instance, shared across all agents

## Common Edit Patterns

### Adding behavior at pipeline end
Find the status assignment in `run()` — look for `result["status"]` or the block where `status` is set based on test results. Insert your logic before the final return dict is built.

### Injecting context into the planner
The planner's `AgentContext` is built in the planning section of `run()`. The `extra` dict is where you pass runtime data. The planner's `build_messages()` reads from `context.extra`. Pattern:
```python
# In controller.py, where planner context is built:
context = AgentContext(
    ...,
    extra={"prelude": self._prelude, "your_new_data": your_data},
)
```
```python
# In planner.py build_messages():
your_data = context.extra.get("your_new_data")
if your_data:
    user_content += f"\n\nYour section:\n{your_data}\n"
```

### Injecting context into the implementer
Same pattern. The implementer's `AgentContext` is built in `_run_implementer()`. It already has `tool_executor`, `symbol_index`, `prelude`, `learned_heuristics`, and `fast_mode` in `extra`.

### Adding a new tool to the implementer
1. Add tool definition dict to `IMPLEMENTER_TOOLS` list (JSON schema format)
2. Add handler in `run()` inside the `for tool_call in response.tool_calls:` dispatch block
3. Handler must append a `{"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": res}` message
4. If the tool writes files, increment `write_count` and add path to `modified_files` or `created_files`
5. If the tool modifies a file, call `symbol_index.invalidate(path)` to refresh the AST cache

### Adding a new agent to the pipeline
1. Create agent class in `glitchlab/agents/` extending `BaseAgent`
2. Add role to `config_loader.py` routing map
3. Add runner function in `runners.py`
4. Wire into `_run_pipeline_step()` or the appropriate pipeline phase in `controller.py`
5. Add model mapping in `glitchlab.yaml`

### Writing failure/status records
GLITCHLAB uses `.glitchlab/` for runtime state. Files go in `.glitchlab/logs/` (audit.jsonl) or directly in `.glitchlab/` (task_state.json). Use `json.dumps()` + append mode for JSONL files. The directory exists by the time the pipeline runs.

### Emitting events
```python
from glitchlab.event_bus import bus
bus.emit(
    event_type="your_event.name",
    payload={"key": "value"},
    agent_id="implementer",  # or whatever role
)
```

## Agent Result Contracts

### Implementer returns:
```python
{
    "changes": [{"file": "path", "action": "modify", "_already_applied": True}],
    "tests_added": [],
    "commit_message": "feat: ...",
    "summary": "What was done",
    "_agent": "implementer",
    "_model": "model-string",
    "_tokens": int,
    "_cost": float,
    "_loop_tokens": int,  # cumulative tokens across all loop steps
}
```

### Debugger returns:
```python
{
    "diagnosis": "What was failing",
    "root_cause": "The specific error",
    "fix": {"changes": [...]},
    "confidence": "high|medium|low",
    "should_retry": True,
    "summary": "What was fixed",
    "_agent": "debugger",
    "_model": "model-string",
    "_tokens": int,
    "_cost": float,
    "_loop_tokens": int,
}
```

### Planner returns:
```python
{
    "steps": [{"step_number": 1, "description": "...", "files": [...], "action": "modify", "code_hint": "...", "do_not_touch": [...]}],
    "files_likely_affected": [...],
    "risk_level": "low|medium|high",
    "requires_core_change": bool,
    "estimated_complexity": "trivial|small|medium|large",
    "_agent": "planner",
    "_model": "model-string",
    "_tokens": int,
    "_cost": float,
}
```

## Things That Break Agents

- `replace_in_file` requires character-perfect `find` strings. Prefer `patch_function` for modifying existing functions.
- `write_file` on existing files risks dropping content. Only use for new files.
- `hard_compact_messages` fires at 50 messages and wipes old context. Call `done` as soon as tests pass.
- `SymbolIndex.invalidate()` may fail silently on tree-sitter version mismatches. This is cosmetic — edits still land.
- The `run_check` tool executor denies commands that aren't in the allowlist. Use `python -m pytest` not bare `pytest`.