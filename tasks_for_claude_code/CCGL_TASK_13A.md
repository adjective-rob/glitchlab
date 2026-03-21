
## Task 13a: Add `PLANNER_TOOLS` list to planner.py

**File:** `glitchlab/agents/planner.py`

**What:** Add the tool definitions list that the planner will use in its agentic loop. Pure addition — no existing code changes.

### Step 1: Insert PLANNER_TOOLS

Find this anchor:

```python
# ---------------------------------------------------------------------------
# Agent Implementation
# ---------------------------------------------------------------------------

class PlannerAgent(BaseAgent):
```

Insert the `PLANNER_TOOLS` list between the comment block and the class definition:

```python
# ---------------------------------------------------------------------------
# Agent Implementation
# ---------------------------------------------------------------------------

PLANNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Reason about the task, analyze what you've read, and draft your plan before submitting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string", "description": "Your analysis of the task and codebase"},
                    "draft_plan": {"type": "string", "description": "Your draft plan outline"}
                },
                "required": ["analysis"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_function",
            "description": "Read the complete body of a function or method. Use this to verify insertion points before including them in your plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Function or method name"},
                    "file": {"type": "string", "description": "Optional file path to restrict search"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_class",
            "description": "Get a class outline showing method signatures with bodies collapsed. Use to understand class structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {"type": "string"},
                    "file": {"type": "string", "description": "Optional file path"}
                },
                "required": ["class_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_grep",
            "description": "Search the codebase for a pattern. Use to find where things are defined or used.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "file_type": {"type": "string", "default": "*.py"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": "Submit your final execution plan as JSON. This ends the planning loop. The JSON must conform to the ExecutionPlan schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_json": {"type": "string", "description": "The complete execution plan as a JSON string"}
                },
                "required": ["plan_json"]
            }
        }
    }
]


class PlannerAgent(BaseAgent):
```

### Do NOT touch

- `PlannerAgent` class, `PlanStep`, `ExecutionPlan`, any existing code.

### Verify

```bash
python -c "from glitchlab.agents.planner import PLANNER_TOOLS; print(f'{len(PLANNER_TOOLS)} tools')"
# Expected: 5 tools
python -c "from glitchlab.agents.planner import PlannerAgent; print('ok')"
python -m pytest tests/ -x
```