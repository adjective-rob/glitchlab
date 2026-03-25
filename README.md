# **⚡ GLITCHLAB** v4.4.0

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

### **Project Structure (v4.3.0)**

`glitchlab/`  
`├── cli.py              # CLI interface (typer)`  
`├── controller.py       # The brainstem`  
`├── router.py           # Vendor-agnostic model routing`  
`├── config_loader.py    # Config loading + merging`  
`├── identity.py         # Versioning and branding`  
`├── agents/             # The roster (Zap, Patch, Reroute, etc.)`  
`├── workspace/          # Git worktree isolation & tools`  
`├── governance/         # Boundary enforcement`  
`└── templates/          # Task and config templates`

---

## **What's New in v4.3.0**

### **Routine Version Bump**

GLITCHLAB v4.3.0 is a minor version bump with no breaking changes. All version strings have been synchronized across the codebase.

### **Previous Release: Zephyr SBOF Integration (v4.2.0)**

GLITCHLAB v4.2.0 introduced **Zephyr SBOF (Signed Bill of Facts)** — cryptographic signing and attestation for every agent action. Every tool call, plan step, and code mutation is now signed with a tamper-evident signature before it is committed to the event log. This gives you:

* **Cryptographic attestation** — each agent action carries a verifiable signature tied to the agent identity and the exact payload it produced.
* **Tamper detection** — any post-hoc modification to an action record is immediately detectable by signature verification.
* **Audit-ready provenance** — the full chain of signed facts can be exported and verified by external tooling, satisfying supply-chain security requirements.

### **EventBus Architecture Upgrade**

The internal EventBus has been upgraded with three new first-class fields on every event:

| Field | Type | Purpose |
| ----- | ----- | ----- |
| `run_id` | `UUID` | Uniquely identifies a single end-to-end agent loop execution |
| `action_id` | `UUID` | Uniquely identifies each discrete action within a run |
| `metadata` | `dict` | Arbitrary structured context (model, token counts, timestamps, etc.) |

Together these fields enable **perfect deterministic traceability** of agent loops: given any event in the log you can reconstruct the exact sequence of actions that produced it, replay the loop with identical inputs, and diff two runs at the action level. The Controller, all agents, and the workspace tooling emit these fields automatically — no configuration required.

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
