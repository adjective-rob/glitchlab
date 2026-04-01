# **⚡ GLITCHLAB** v4.5.0

**The Agentic Dev Engine — Build Weird. Ship Clean.**  
 A local, repo-agnostic, multi-agent development engine that evolves codebases under strict governance.

---

## **What It Does**

GLITCHLAB takes a development task (GitHub issue, local YAML, or interactive prompt), breaks it into an execution plan, implements changes, runs tests, fixes failures, scans for security issues, and opens a PR—all orchestrated locally with deterministic control.

---

## **Agent Roster**

| Agent | Role | Model | Energy |
| ----- | ----- | ----- | ----- |
| 🧠 **Professor Zap** | Planner | Gemini | Manic genius with whiteboard chaos |
| 🔧 **Patch** | Implementer | Claude | Hoodie-wearing prodigy |
| 🐛 **Reroute** | Debugger | Claude | Quiet gremlin (appears when things break) |
| 🔒 **Firewall Frankie** | Security | Gemini | Cartoon cop with magnifying glass |
| 📦 **Semver Sam** | Release | Gemini | Accountant with neon sneakers |
| 🛡️ **Shield** | TestGen | Gemini | Methodical guardian of test coverage |
| 🗄️ **Archivist Nova** | Archivist | Gemini | Keeper of context and project memory |

---

## **Quick Start**

### **1\. Install**

`cd glitchlab`  
`pip install -e .`

### **2\. Configure API Keys**

`cp .env.example .env`  
`# Edit .env with your keys:`  
`# ANTHROPIC_API_KEY=sk-ant-...`  
`# GOOGLE_API_KEY=AI...`

### **3\. Initialize a Repository**

`cd ~/your-project`  
`glitchlab init  # Bootstraps .glitchlab (defaults to current directory)`

### **4\. Run**

From a GitHub issue:

`glitchlab run --repo . --issue 42`

From a local task file:

`glitchlab run --repo . --local-task`

Interactive mode:

`glitchlab interactive --repo .`


#### Surgical Mode

Run the pipeline in surgical mode to apply a minimal, focused fix using a specialized surgical pipeline profile. Use the `--surgical` flag with the `run` or `interactive` commands:

```bash
glitchlab run --repo . --issue 42 --surgical
```

or

```bash
glitchlab interactive --repo . --surgical
```

This mode loads the `surgical` profile configuration, limits fix attempts to one, and skips the planning step for a streamlined fix process.

### **5\. Check Status**

`glitchlab status --repo .`

---

## **Task Sources**

* **GitHub Issues:** Label issues with `glitchlab` and use the provided issue template.

* **Local YAML Tasks:** Create files under `.glitchlab/tasks/queue/next.yaml`.

* **Interactive:** Just describe what you want. GLITCHLAB plans, you approve, it executes.

---

## **Human Intervention Points**

GLITCHLAB is autonomous between checkpoints, but you stay in control:

* **Plan Review:** Approve before implementation begins.

* **Core Boundary:** Use `--allow-core` for protected paths.

* **Fix Loop:** Halts after a set number of failed attempts and asks what to do next.

* **Pre-PR Review:** View the diff and approve or cancel before the PR opens.

* **Budget Cap:** Halts if a token/dollar limit is exceeded.

---

## **Architecture**

The **Controller** is the brainstem. It never writes code directly; it only coordinates.

### **Project Structure (v4.5.0)**

`glitchlab/`  
`├── cli.py              # CLI interface (typer)`  
`├── controller.py       # The brainstem (thin orchestration shell)`  
`├── step_handlers.py    # Per-agent post-processing (registry pattern)`  
`├── agent_runners.py    # Context builders + agent invocations`  
`├── lifecycle.py        # Startup, finalize, PR, session`  
`├── router.py           # Vendor-agnostic model routing`  
`├── config_loader.py    # Config loading + profile merging`  
`├── scope.py            # Computed context via dependency analysis`  
`├── symbols.py          # AST-aware symbol index (tree-sitter)`  
`├── indexer.py          # Repo navigator (route map)`  
`├── prelude.py          # Prelude integration (codebase memory)`  
`├── brain_writer.py     # Persistent learned heuristics`  
`├── history.py          # Append-only run log + pattern extraction`  
`├── context_compressor.py # Token budget management`  
`├── task_quality.py     # Ambiguity detection gate`  
`├── identity.py         # Versioning and branding`  
`├── agents/             # The roster (Zap, Patch, Reroute, etc.)`  
`├── workspace/          # Git worktree isolation & tools`  
`├── auditor/            # Scanner + task writer (ouroboros)`  
`├── governance/         # Boundary enforcement + budget control`  
`├── reporting/          # Run reports and summaries`  
`└── profiles/           # Pipeline profiles (default, surgical)`

---

## **How It Works**

GlitchLab is a pipeline. A task enters one end, a tested PR exits the other. Between those two points, seven agents operate in sequence — each one does exactly one job, then hands off to the next. The Controller coordinates but never writes code. It's a brainstem, not a brain.

### **The Pipeline**

```
Task → 🧠 Plan → 🔧 Implement → 🐛 Debug → 🛡️ Test → 🔒 Secure → 📦 Release → 📚 Archive → PR
```

Every step is a gate. If something fails, the pipeline halts, asks you what to do, and waits. Nothing ships without your approval.

### **The Agents**

**🧠 Professor Zap** — *The Planner*
Reads the task, reads the codebase, and produces a step-by-step execution plan. Identifies which files will change, what the risks are, and what not to touch. Never writes code — only thinks. Plans are capped at 4 steps. If Zap can't fit the work into 4 steps, the task is too big and gets rejected.

**🔧 Patch** — *The Implementer*
The only agent that writes code. Operates in a tool loop: think, read, search, write, verify. Has a circuit breaker — if Patch reaches step 10 without writing anything useful, the run aborts instead of burning tokens on exploration. Reads are scaled to the plan scope — bigger plans get more context budget. After every write, Patch must verify the change before declaring done.

**🐛 Reroute** — *The Debugger*
Appears when tests fail. Runs a fix loop: read the error, think about the root cause, apply a fix, re-run tests. Scope-locked — Reroute only fixes tests broken by the current change, not pre-existing failures. Has a thrash detector — if it tries the same fix twice, it stops. Budget scales with task complexity. If failures cascade (more tests breaking with each fix attempt), Reroute aborts early.

**🛡️ Shield** — *The Test Generator*
A single-shot agent. After Patch writes code, Shield writes one focused regression test to verify the change does what it's supposed to do. Runs automatically — no configuration needed.

**🔒 Firewall Frankie** — *The Security Scanner*
Scans the diff for dangerous patterns: hardcoded secrets, unsafe dependencies, permission escalations, injection risks. Can issue a warning (continue with caution) or a block (stops the pipeline). Blocks require explicit human override.

**📦 Semver Sam** — *The Release Guardian*
Analyzes the API surface to determine version bump semantics. Writes the changelog entry. Decides if this is a patch, minor, or major bump. Energy of an accountant who wears neon sneakers.

**📚 Archivist Nova** — *The Documentarian*
Captures design decisions after the work is done. Writes Architecture Decision Records when the change is significant enough to warrant one. Keeps future-you from asking "why did we do it this way?"

### **The Infrastructure**

**The Controller** — Thin orchestration shell. Pulls a task, builds a RunContext (the shared state bundle), iterates pipeline steps, enforces stop conditions. Dispatches everything, decides nothing. All post-processing logic lives in a handler registry — add a new agent type by writing one function and registering it. No controller edits required.

**The Router** — Model-agnostic dispatch. Agents request completions by role, the router sends them to the configured model. Swap models in `config.yaml` without changing agent code. Tracks token usage and cost per agent. Enforces budget caps — if a run exceeds its dollar limit, the pipeline halts.

**The Scope Resolver** — Computes file context through actual dependency analysis. When Patch needs to modify `controller.py`, the scope resolver traces its imports, pulls signature summaries of dependencies, and provides exactly the context needed — not a full repo dump, not a guess.

**The Symbol Index** — AST-level code intelligence via tree-sitter. Find references, extract function bodies, get class outlines. Agents use this for surgical navigation instead of grepping through files.

**Prelude** — The memory. Generates and maintains structured context about the codebase — stack, architecture, patterns, constraints, decisions. Agents consume this at the start of every run so they understand the project before they plan or write anything. Prelude is the reason GlitchLab doesn't start cold every time.

**The Brain** — Persistent learned heuristics from prior runs. Tracks which files need to be read together, which edit strategies succeed, which patterns fail. Accumulates over time. A codebase with 100 runs of brain data produces better agent performance than one with zero — the brain tells agents what worked before.

**The Auditor** — The ouroboros. Scans the codebase with tree-sitter (no API calls), finds actionable improvements (missing docs, TODOs, complex functions, untested code), and generates well-scoped GlitchLab task files. GlitchLab generates work for itself.

### **Governance**

**Boundaries** — Protected paths that agents cannot modify without explicit `--allow-core` override. Prevents autonomous changes to critical infrastructure.

**Budget Controls** — Per-task token limits and dollar caps. The pipeline halts if either is exceeded. You never wake up to a surprise API bill.

**Quality Gate** — Catches vague or ambiguous task objectives before they enter the pipeline. If the objective says "clean up" or "improve," the gate injects narrow-interpretation constraints that prevent the implementer from redesigning half the codebase.

**Event Bus + Zephyr** — Every action is a signed event. The event bus broadcasts to subscribers (audit logger, Zephyr signer). The audit log is append-only JSONL with cryptographic attestation. Every tool call, every plan step, every code mutation is traceable, verifiable, and tamper-evident.

---

## **What's New in v4.5.0**

### **Pipeline Reliability Hardening (v4.4.0–v4.5.0)**

The pipeline internals received major reliability improvements across every agent:

- **Write-deadline circuit breaker** — implementer aborts after step 10 if no meaningful write has occurred, preventing token-burning exploration spirals (#134).
- **Debugger thrash detection** — detects and stops repeated failed fix attempts instead of looping to exhaustion (#135).
- **Scope-locked debugger** — debugger only fixes tests broken by the current change, ignoring pre-existing failures (#136).
- **Task quality gate** — rejects vague auditor objectives before they enter the pipeline, preventing ambiguous tasks from causing implementer exploration (#137).
- **Dynamic debugger budgets** — fix attempt limits scale with task complexity instead of using a fixed cap (#138).
- **Code hint validation** — planner output is checked for specificity; vague hints are flagged before reaching the implementer (#139).
- **Cascading failure abort** — debug loop exits early when test failures are increasing rather than decreasing (#141).
- **Planner step cap** — plans exceeding 4 steps are rejected and re-planned to prevent overscoped implementations (#144).
- **Auto-surgical detection** — batch mode tasks that are trivially scoped automatically run in surgical mode (#145).
- **Implementer read scaling** — read cap adjusts based on plan scope so larger tasks get more context budget (#152).
- **Empty PR prevention** — skips PR creation when the implementer produces zero real file changes (#153).

### **Auditor & Governance**

- **Protected file gate** — auditor task writer cannot generate tasks targeting protected paths (#156).
- **Task decomposition rules** — auditor embeds scope constraints into generated tasks to prevent oversized work items (#142).
- **Implementer done gate** — implementer must run verification check after writes before declaring done (#143).

### **Prelude Integration**

- Minimum Prelude version updated to 1.4.0 (compact context injection requires it).
- Prelude 1.5.0 now available with MCP server support for AI tool integration.

### **Previous: Zephyr SBOF Integration (v4.2.0)**

Cryptographic signing and attestation for every agent action via Zephyr SBOF. Every tool call, plan step, and code mutation is signed with a tamper-evident signature.

---

## **Design Principles**

* **Local-first:** Runs on your machine.

* **Repo-agnostic:** Works with Rust, Python, TypeScript, Go—anything.

* **Deterministic:** Controller logic is explicit, not probabilistic.

* **Bounded:** Budget caps, retry limits, and tool allowlists keep operations contained.

* **Under 2k lines:** Maintainable, focused, and fast.

---

## **License**

MIT — Adjective LLC
