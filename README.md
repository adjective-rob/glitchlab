# ⚡ GLITCHLAB

**The Agentic Dev Engine — Build Weird. Ship Clean.**

A local, repo-agnostic, multi-agent development engine that evolves codebases under strict governance.

## What It Does

GLITCHLAB takes a development task (GitHub issue, local YAML, or interactive prompt), breaks it into an execution plan, implements the changes, runs tests, fixes failures, scans for security issues, and opens a PR — all orchestrated locally with deterministic control.

## Agent Roster

| Agent | Role | Model | Energy |
|-------|------|-------|--------|
| 🧠 Professor Zap | Planner | Gemini | Manic genius with whiteboard chaos |
| 🔧 Patch | Implementer | Claude | Hoodie-wearing prodigy |
| 🐛 Reroute | Debugger | Claude | Quiet gremlin (appears when things break) |
| 🔒 Firewall Frankie | Security | Gemini | Cartoon cop with magnifying glass |
| 📦 Semver Sam | Release | Gemini | Accountant with neon sneakers |

## Quick Start

### 1. Install

```bash
cd glitchlab
pip install -e .
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   GOOGLE_API_KEY=AI...
```

### 3. Initialize a Repository

```bash
glitchlab init ~/your-project
```

### 4. Run

**From GitHub issue:**
```bash
glitchlab run --repo ~/your-project --issue 42
```

**From local task file:**
```bash
glitchlab run --repo ~/your-project --local-task
```

**Interactive mode:**
```bash
glitchlab interactive --repo ~/your-project
```

### 5. Check Status

```bash
glitchlab status --repo ~/your-project
```

## Task Sources

### GitHub Issues
Label issues with `glitchlab`. Use the provided issue template.

### Local YAML Tasks
Create `.glitchlab/tasks/next.yaml`:

```yaml
id: my-task-001
objective: "Add --json flag to CLI output"
constraints:
  - "No new dependencies"
  - "Must not modify core modules"
acceptance:
  - "Tests pass"
  - "New test added"
risk: low
```

### Interactive
Just describe what you want. GLITCHLAB plans, you approve, it executes.

## Human Intervention Points

GLITCHLAB is autonomous between checkpoints, but you stay in control:

1. **Plan Review** — Approve before implementation begins
2. **Core Boundary** — `--allow-core` required for protected paths
3. **Fix Loop** — Halts after N failed attempts, asks what to do
4. **Pre-PR Review** — See the diff, approve or cancel
5. **Budget Cap** — Halts if token/dollar limit exceeded

## Configuration

Per-repo overrides in `.glitchlab/config.yaml`:

```yaml
routing:
  implementer: "anthropic/claude-sonnet-4-20250514"

boundaries:
  protected_paths:
    - "crates/zephyr-core"

limits:
  max_fix_attempts: 3
  max_dollars_per_task: 5.0
```

## Architecture

```
Backlog Source (GitHub / Local / Interactive)
         │
         ▼
    Controller (deterministic orchestrator)
         │
    ┌────┼────┐
    ▼    ▼    ▼
  Plan  Impl  Debug ──→ Fix Loop
         │
         ▼
    Security + Release
         │
         ▼
      PR Creation
```

The Controller is the brainstem. It never writes code. It only coordinates.

## Cost Model

You only pay for API tokens. The controller runs on your laptop. No GPU, no cloud, no infra.

Default budget: **$2–$10 per task**, max 4 fix attempts.

## Project Structure

```
glitchlab/
├── cli.py              # CLI interface (typer)
├── controller.py       # The brainstem
├── router.py           # Vendor-agnostic model routing
├── config.yaml         # Default configuration
├── config_loader.py    # Config loading + merging
├── agents/
│   ├── planner.py      # Professor Zap
│   ├── implementer.py  # Patch
│   ├── debugger.py     # Reroute
│   ├── security.py     # Firewall Frankie
│   └── release.py      # Semver Sam
├── workspace/
│   ├── __init__.py     # Git worktree isolation
│   └── tools.py        # Safe command execution
└── governance/
    └── __init__.py     # Boundary enforcement
```

## Design Principles

- **Local-first** — runs on your machine
- **Repo-agnostic** — works with Rust, Python, TS, Go, anything
- **Vendor-agnostic** — LiteLLM abstracts model providers
- **Deterministic** — controller logic is explicit, not ML
- **Bounded** — budget caps, retry limits, tool allowlists
- **Under 2k lines** — if it grows beyond that, you've overbuilt

## License

MIT — Adjective LLC


## Versioning

Check your current version using:
```bash
glitchlab --version
```