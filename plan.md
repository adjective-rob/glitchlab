# YAML Agent Messaging — Implementation Plan

## Problem Statement

Currently, glitchlab agents communicate with LLMs using two formats:
1. **Free-form text** — Most agents (implementer, debugger, security, archivist, release) construct user messages as unstructured prose with embedded data (file lists, plan steps, diffs, etc.)
2. **JSON** — Planner and TestGen force `response_format: json_object` and expect raw JSON back, which they parse with `json.loads()`

Both approaches have cost and accuracy problems:
- **Prose messages** are verbose, wasting tokens on filler words and inconsistent formatting
- **JSON responses** are token-expensive (every key is quoted, braces/commas add up) and fragile (truncated JSON is unrecoverable)
- **Inter-agent state** (`TaskState.to_agent_summary()`) returns Python dicts that get interpolated as str(dict) in message templates — ugly and inconsistent

## Why YAML

- ~30-40% fewer tokens than equivalent JSON (no quotes on keys, no braces, no commas)
- Graceful degradation: truncated YAML is still partially parseable (unlike JSON)
- Human-readable by default — easier to debug agent transcripts
- Python's `yaml` library is already a dependency (used in config_loader.py)
- LLMs are well-trained on YAML output and produce it reliably

## Architecture

There are **three messaging surfaces** to convert:

### Surface 1: Agent → LLM (Input Messages)
How agents format the user prompt sent to the LLM.

**Current:** Free-form f-strings with inconsistent structure
**Target:** YAML-formatted structured payloads in user messages

Files: All `build_messages()` methods in:
- `glitchlab/agents/planner.py`
- `glitchlab/agents/implementer.py`
- `glitchlab/agents/debugger.py`
- `glitchlab/agents/security.py`
- `glitchlab/agents/release.py`
- `glitchlab/agents/archivist.py`
- `glitchlab/agents/testgen.py`

### Surface 2: LLM → Agent (Response Parsing)
How agents parse structured responses from the LLM.

**Current:** `json.loads()` + Pydantic in planner.py, testgen.py; tool-call JSON args in others
**Target:** `yaml.safe_load()` for single-shot agents (planner, testgen); tool-call agents stay as-is (tool args are JSON per the OpenAI spec — this is non-negotiable)

Files:
- `glitchlab/agents/planner.py` — `parse_response()` + system prompt schema example
- `glitchlab/agents/testgen.py` — `parse_response()` + system prompt schema example

### Surface 3: Controller → Agent (Inter-Agent State)
How `TaskState.to_agent_summary()` formats data passed between agents.

**Current:** Returns raw Python dicts, interpolated as `str(dict)` or accessed via `.get()`
**Target:** `yaml.dump()` the summary dict so agents receive clean YAML blocks

Files:
- `glitchlab/controller.py` — `TaskState.to_agent_summary()` + `TaskState.persist()`
- `glitchlab/agents/__init__.py` — Add YAML helper to `BaseAgent`

## Implementation Steps

### Step 1: Add YAML formatting helpers to BaseAgent
Add `_yaml_block(data)` utility to `BaseAgent` in `glitchlab/agents/__init__.py` that wraps `yaml.dump()` with consistent settings (default_flow_style=False, sort_keys=False).

### Step 2: Convert planner.py to YAML I/O
- Change system prompt to show YAML output schema instead of JSON
- Change `build_messages()` to format user content as YAML
- Change `parse_response()` to use `yaml.safe_load()` instead of `json.loads()`
- Remove `response_format: json_object` from `run()` (YAML is not a supported response_format)

### Step 3: Convert testgen.py to YAML I/O
- Same changes as planner: system prompt, build_messages, parse_response
- Remove `response_format: json_object`

### Step 4: Convert tool-loop agents' build_messages to YAML input
For implementer, debugger, security, release, archivist:
- Rewrite `build_messages()` to format the user content as a YAML document
- System prompts stay as prose (they are instructions, not data)
- Tool call arguments stay as JSON (API requirement)

### Step 5: Convert TaskState.to_agent_summary() to emit YAML
- Add a `to_yaml_summary(for_agent)` method that returns a YAML string
- Update controller to pass YAML summaries into `AgentContext.previous_output`

### Step 6: Update TaskState.persist() to write YAML
- Change from `model_dump_json()` to `yaml.dump(model_dump())`
- Rename output file from `task_state.json` to `task_state.yaml`

### Step 7: Update tests
- Update any tests that assert on JSON format to expect YAML
- Verify parsing round-trips

## What NOT to change
- Tool definitions (IMPLEMENTER_TOOLS, DEBUGGER_TOOLS, etc.) — these are OpenAI function-calling schemas and must remain JSON
- Tool call argument parsing (`json.loads(tool_call.function.arguments)`) — API returns JSON
- EventBus/GlitchEvent — internal pub/sub, not agent messaging
- Router — transport layer, format-agnostic
- Config loading — already uses YAML
