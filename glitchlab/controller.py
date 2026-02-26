"""
GLITCHLAB Controller — The Brainstem (v2)

The most important piece. It is NOT smart. It is deterministic.

v2 Architecture: Context Router (not Context Reservoir)
  - Layer 0: Immutable system contracts per agent
  - Layer 1: ScopeResolver computes precise file/symbol context
  - Layer 2: Tool-centric retrieval (agents pull, not pushed)
  - Layer 3: Structured TaskState flows between agents

Responsibilities:
  - Pull next task
  - Create isolated worktree
  - Maintain TaskState across agent steps
  - Assign agent roles in order
  - Track token budget
  - Track retry attempts
  - Enforce stop conditions
  - Open PR
  - Clean up workspace

It never writes code. It only coordinates.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Any
from pydantic import BaseModel, Field, model_validator

import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table

from glitchlab.agents import AgentContext
from glitchlab.agents.archivist import ArchivistAgent
from glitchlab.agents.debugger import DebuggerAgent
from glitchlab.agents.implementer import ImplementerAgent
from glitchlab.agents.planner import PlannerAgent
from glitchlab.agents.release import ReleaseAgent
from glitchlab.agents.security import SecurityAgent
from glitchlab.config_loader import GlitchLabConfig, load_config
from glitchlab.governance import BoundaryEnforcer, BoundaryViolation
from glitchlab.history import TaskHistory
from glitchlab.indexer import build_index
from glitchlab.prelude import PreludeContext
from glitchlab.router import BudgetExceededError, Router
from glitchlab.workspace import Workspace
from glitchlab.workspace.tools import ToolExecutor, ToolViolationError

console = Console()


# ---------------------------------------------------------------------------
# Task State — Structured Working Memory (Layer 3)
# ---------------------------------------------------------------------------

class StepState(BaseModel):
    """Tracks the status of an individual planned step."""
    step_number: int
    description: str = ""
    files: list[str] = Field(default_factory=list)
    action: str = ""
    status: Literal["pending", "completed", "failed", "skipped"] = "pending"
    outcome: str = ""


class TaskState(BaseModel):
    """
    Structured working memory that flows between agents.

    Replaces the old pattern of passing raw `previous_output` blobs.
    Each agent reads what it needs and writes its contribution.
    The Controller owns this object and persists it per-run.
    """

    task_id: str
    objective: str
    mode: str = "evolution"
    risk_level: str = "low"

    # Planner output (consumed by Implementer, Debugger, Security)
    plan_steps: list[StepState] = Field(default_factory=list)
    files_in_scope: list[str] = Field(default_factory=list)
    estimated_complexity: str = "medium"
    requires_core_change: bool = False

    # Implementer output (consumed by Debugger, Security, Release)
    files_modified: list[str] = Field(default_factory=list)
    files_created: list[str] = Field(default_factory=list)
    tests_added: list[str] = Field(default_factory=list)
    commit_message: str = ""
    implementation_summary: str = ""

    # Debug loop state
    test_passing: bool = False
    debug_attempts: int = 0
    last_error: str = ""
    previous_fixes: list[dict] = Field(default_factory=list)

    # Security + Release
    security_verdict: str = ""
    version_bump: str = ""
    changelog_entry: str = ""

    # Tracking
    completed_phases: list[str] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)

    def mark_phase(self, phase: str) -> None:
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)

    def to_agent_summary(self, for_agent: str) -> dict:
        """
        Return only the fields relevant to a specific agent.
        This is the core of the context-router pattern: agents get
        precisely what they need, not everything.
        """
        base = {
            "task_id": self.task_id,
            "objective": self.objective,
            "mode": self.mode,
            "risk_level": self.risk_level,
        }

        if for_agent == "planner":
            # Planner gets task + history of what failed before
            return {
                **base,
                "previous_fixes": self.previous_fixes[-3:] if self.previous_fixes else [],
            }

        elif for_agent == "implementer":
            return {
                **base,
                "plan_steps": [s.model_dump() for s in self.plan_steps],
                "files_in_scope": self.files_in_scope,
                "estimated_complexity": self.estimated_complexity,
            }

        elif for_agent == "debugger":
            return {
                **base,
                "files_modified": self.files_modified,
                "files_created": self.files_created,
                "last_error": self.last_error,
                "debug_attempts": self.debug_attempts,
                "previous_fixes": self.previous_fixes[-2:] if self.previous_fixes else [],
            }

        elif for_agent == "security":
            return {
                **base,
                "files_modified": self.files_modified,
                "files_created": self.files_created,
                "implementation_summary": self.implementation_summary,
            }

        elif for_agent == "release":
            return {
                **base,
                "files_modified": self.files_modified,
                "implementation_summary": self.implementation_summary,
                "security_verdict": self.security_verdict,
            }

        elif for_agent == "archivist":
            return {
                **base,
                "plan_steps": [s.model_dump() for s in self.plan_steps],
                "files_modified": self.files_modified,
                "implementation_summary": self.implementation_summary,
                "version_bump": self.version_bump,
            }

        return base

    def persist(self, ws_path: Path) -> None:
        """Write current state to workspace for debugging/auditing."""
        state_dir = ws_path / ".glitchlab"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "task_state.json").write_text(
            self.model_dump_json(indent=2)
        )

class DirtyRepoError(Exception):
    """Raised when the main repository has uncommitted changes."""
    pass

# ---------------------------------------------------------------------------
# Scope Resolver (Layer 1) — Computed Context, Not Guessed
# ---------------------------------------------------------------------------

class ScopeResolver:
    """
    Computes precise file context for agents based on actual
    dependency analysis rather than planner guesses.

    Replaces the old `gather_file_context` pattern that blindly
    read whatever the planner listed.
    """

    def __init__(self, working_dir: Path, repo_index: Any = None):
        self.working_dir = working_dir
        self.repo_index = repo_index

    def resolve_for_files(
        self,
        target_files: list[str],
        max_lines: int = 2000,
        include_deps: bool = True,
    ) -> dict[str, str]:
        """
        Read target files + optionally resolve their local imports
        to provide dependency signatures.
        """
        context = {}

        for fpath in target_files:
            full = self.working_dir / fpath
            if not full.exists() or not full.is_file():
                continue

            try:
                lines = full.read_text().splitlines()
                if len(lines) > max_lines:
                    content = "\n".join(lines[:max_lines]) + f"\n\n... truncated ({len(lines)} lines total)"
                else:
                    content = "\n".join(lines)
                context[fpath] = content
            except Exception as e:
                context[fpath] = f"(could not read: {e})"

            # Resolve local dependencies and include signatures only
            if include_deps:
                deps = self._resolve_imports(full)
                for dep_path, signatures in deps.items():
                    if dep_path not in context:
                        context[f"[dep] {dep_path}"] = signatures

        return context

    def _resolve_imports(self, file_path: Path) -> dict[str, str]:
        """
        Parse imports from a file and return signature summaries
        of local dependencies (not full file contents).
        """
        deps = {}
        try:
            content = file_path.read_text()
        except Exception:
            return deps

        suffix = file_path.suffix

        if suffix == ".py":
            deps = self._resolve_python_imports(content, file_path)
        elif suffix == ".rs":
            deps = self._resolve_rust_imports(content, file_path)
        elif suffix in (".ts", ".tsx", ".js", ".jsx"):
            deps = self._resolve_js_imports(content, file_path)

        return deps

    def _resolve_python_imports(self, content: str, source: Path) -> dict[str, str]:
        """Extract local Python imports and return their signatures."""
        deps = {}
        for line in content.splitlines():
            line = line.strip()

            # Match: from glitchlab.foo import Bar
            match = re.match(r'^from\s+(glitchlab\.\S+)\s+import', line)
            if match:
                module = match.group(1).replace(".", "/") + ".py"
                dep_path = self.working_dir / module
                if dep_path.exists():
                    sigs = self._extract_python_signatures(dep_path)
                    if sigs:
                        deps[module] = sigs

        return deps

    def _resolve_rust_imports(self, content: str, source: Path) -> dict[str, str]:
        """Extract local Rust use statements and return signatures."""
        deps = {}
        for line in content.splitlines():
            match = re.match(r'^use\s+crate::(\S+)', line.strip())
            if match:
                mod_path = match.group(1).replace("::", "/")
                # Try both mod.rs and direct .rs
                for candidate in [
                    self.working_dir / "src" / f"{mod_path}.rs",
                    self.working_dir / "src" / mod_path / "mod.rs",
                ]:
                    if candidate.exists():
                        sigs = self._extract_rust_signatures(candidate)
                        if sigs:
                            deps[str(candidate.relative_to(self.working_dir))] = sigs
                        break
        return deps

    def _resolve_js_imports(self, content: str, source: Path) -> dict[str, str]:
        """Extract local JS/TS imports and return signatures."""
        deps = {}
        for line in content.splitlines():
            match = re.match(r'''(?:import|from)\s+[^'"]*['"](\./[^'"]+)['"]''', line.strip())
            if match:
                rel = match.group(1)
                for ext in ["", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js"]:
                    candidate = (source.parent / (rel + ext)).resolve()
                    if candidate.exists() and candidate.is_file():
                        sigs = self._extract_js_signatures(candidate)
                        if sigs:
                            deps[str(candidate.relative_to(self.working_dir))] = sigs
                        break
        return deps

    @staticmethod
    def _extract_python_signatures(path: Path) -> str:
        """Extract class/function signatures from a Python file."""
        lines = []
        try:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith(("class ", "def ", "async def ")):
                    lines.append(stripped.split(":")[0] + ":")
                elif stripped.startswith('"""') and lines:
                    lines.append(f"    {stripped}")
        except Exception:
            pass
        return "\n".join(lines) if lines else ""

    @staticmethod
    def _extract_rust_signatures(path: Path) -> str:
        """Extract pub fn/struct/enum signatures from a Rust file."""
        lines = []
        try:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith(("pub fn ", "pub async fn ", "pub struct ", "pub enum ", "pub trait ")):
                    lines.append(stripped.rstrip("{").strip())
        except Exception:
            pass
        return "\n".join(lines) if lines else ""

    @staticmethod
    def _extract_js_signatures(path: Path) -> str:
        """Extract export signatures from a JS/TS file."""
        lines = []
        try:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith(("export ", "export default ")):
                    # Grab just the signature line
                    sig = stripped.split("{")[0].strip()
                    if sig:
                        lines.append(sig)
        except Exception:
            pass
        return "\n".join(lines) if lines else ""


# ---------------------------------------------------------------------------
# Task Definition
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """Represents a single unit of work for GLITCHLAB."""
    task_id: str = Field(..., alias="id", description="Unique ID for the task")
    objective: str = Field(..., description="The main objective to complete")
    constraints: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(
        default_factory=lambda: ["Tests pass", "Clean diff"],
        alias="acceptance",
    )
    risk_level: Literal["low", "medium", "high"] = Field(default="low", alias="risk")
    source: str = Field(default="local")
    mode: Literal["maintenance", "evolution"] | None = Field(default=None)
    file_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode='after')
    def determine_mode(self) -> "Task":
        if not self.mode:
            if self.risk_level == "low" and any(
                term in self.objective.lower()
                for term in ["doc", "lint", "format", "fix"]
            ):
                self.mode = "maintenance"
            else:
                self.mode = "evolution"
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "Task":
        with open(path) as f:
            data = yaml.safe_load(f)
        data["task_id"] = data.get("id", path.stem)
        data["source"] = "local-file"
        data["file_path"] = path
        return cls(**data)

    @classmethod
    def from_interactive(cls, objective: str) -> "Task":
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return cls(
            task_id=f"interactive-{ts}",
            objective=objective,
            source="interactive",
        )

    @classmethod
    def from_github_issue(cls, repo_path: Path, issue_number: int) -> "Task":
        """Fetch issue from GitHub CLI."""
        result = subprocess.run(
            ["gh", "issue", "view", str(issue_number), "--json", "title,body,labels,number"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to fetch issue #{issue_number}: {result.stderr}")

        data = json.loads(result.stdout)
        labels = [lbl["name"] for lbl in data.get("labels", [])]
        mode = "maintenance" if "maintenance" in labels else ("evolution" if "evolution" in labels else None)

        return cls(
            task_id=f"gh-{issue_number}",
            objective=f"{data['title']}\n\n{data.get('body', '')}",
            risk_level="high" if "core" in labels else "low",
            source="github",
            mode=mode,
        )


# ---------------------------------------------------------------------------
# Change Applicator (supports full content + unified diffs)
# ---------------------------------------------------------------------------

def _looks_like_diff(text: str) -> bool:
    """Check if text looks like a unified diff vs plain file content."""
    lines = text.strip().split("\n")[:20]
    diff_markers = 0
    for line in lines:
        if line.startswith(("---", "+++", "@@", "diff ")):
            diff_markers += 1
        if line.startswith(("--- a/", "+++ b/", "diff --git")):
            return True
    return diff_markers >= 2


def _normalize_change(change: dict) -> dict:
    """
    Normalize an LLM-produced change dict so that content is always available.

    LLMs frequently put full file content in the 'patch' field instead of 'content'.
    This detects that case and promotes 'patch' to 'content'.
    """
    patch = change.get("patch")
    content = change.get("content")

    if patch and patch.strip() and not content:
        if not _looks_like_diff(patch):
            logger.info(
                f"[NORMALIZE] 'patch' field for {change.get('file', '?')} is not a diff — "
                "promoting to 'content'"
            )
            change["content"] = patch
            change["patch"] = None

    # Strip markdown fences from either field
    for field in ("content", "patch"):
        val = change.get(field)
        if val and val.strip().startswith("```"):
            lines = val.strip().split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            change[field] = "\n".join(lines)

    return change


def apply_changes(
    working_dir: Path,
    changes: list[dict],
    boundary: BoundaryEnforcer | None = None,
    allow_core: bool = False,
    allow_test_modifications: bool = False,
    allow_full_rewrite: bool = True,
) -> list[str]:
    """
    Apply implementation changes using Surgical Blocks or Full Content.
    """
    applied = []
    for change in changes:
        filename = change.get("file", "")
        if not filename:
            continue

        if boundary:
            boundary.check([filename], allow_core)

        fpath = working_dir / filename
        action = change.get("action", "modify")
        
        # New v2.1 Logic: Extract Surgical Blocks
        surgical_blocks = change.get("surgical_blocks", [])
        full_content = change.get("content")

        if action == "create":
            if not full_content:
                applied.append(f"FAIL {filename} (creation requires content)")
                continue
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(full_content)
            applied.append(f"CREATE {filename}")

        elif action == "delete":
            if fpath.exists():
                fpath.unlink()
                applied.append(f"DELETE {filename}")

        elif action == "modify":
            if not fpath.exists():
                applied.append(f"FAIL {filename} (file not found for modification)")
                continue

            current_text = fpath.read_text()

            # ── Strategy 1: Surgical Blocks ──
            if surgical_blocks:
                success_count = 0
                temp_text = current_text
                
                for block in surgical_blocks:
                    search_str = block.get("search", "")
                    replace_str = block.get("replace", "")
                    
                    if search_str and search_str in temp_text:
                        temp_text = temp_text.replace(search_str, replace_str)
                        success_count += 1
                    else:
                        logger.warning(f"[APPLY] Surgical block search failed in {filename}")

                if success_count == len(surgical_blocks):
                    fpath.write_text(temp_text)
                    applied.append(f"SURGICAL {filename} ({success_count} blocks)")
                    continue
                else:
                    logger.warning(f"[APPLY] Some blocks failed in {filename}. Falling back...")

            # ── Strategy 2: Full Content Fallback ──
            if full_content:
                if not allow_full_rewrite:
                    applied.append(f"FAIL {filename} (full rewrite blocked in maintenance mode)")
                else:
                    fpath.write_text(full_content)
                    applied.append(f"MODIFY {filename} (full content)")
            else:
                applied.append(f"FAIL {filename} (no valid surgical blocks or content)")

    return applied


def apply_tests(
    working_dir: Path,
    tests: list[dict],
    allow_test_modifications: bool = False,
) -> list[str]:
    """Apply test file changes with explicit permission check."""
    if tests and not allow_test_modifications:
        raise BoundaryViolation("Test mutation blocked by current governance mode.")

    applied = []
    for test in tests:
        fpath = working_dir / test["file"]
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(test.get("content", ""))
        applied.append(f"TEST {test['file']}")
    return applied


def _apply_patch(working_dir: Path, patch: str) -> bool | str:
    """Apply a unified diff using the 'patch' CLI."""
    logger.debug(f"[PATCH] Raw patch content:\n{patch[:1000]}")

    cleaned = patch.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    if not any(line.startswith(("---", "diff ", "@@")) for line in cleaned.split("\n")):
        msg = "Not a valid unified diff (missing ---, diff, or @@ markers)"
        logger.warning(f"[PATCH] {msg}")
        return msg

    patch_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", dir=working_dir, delete=False
        ) as f:
            f.write(cleaned)
            patch_file = f.name

        result = subprocess.run(
            ["patch", "-p1", "--force", "--fuzz=3", "-i", patch_file],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            return True

        error = (result.stderr or result.stdout).strip()
        logger.warning(f"[PATCH] patch failed: {error}")
        return error

    except Exception as e:
        logger.warning(f"[PATCH] Exception applying patch: {e}")
        return str(e)
    finally:
        if patch_file:
            try:
                Path(patch_file).unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Surgical Doc Comment Insertion
# ---------------------------------------------------------------------------

def insert_doc_comments(file_path: Path, router: Any) -> bool:
    """
    Surgically insert /// doc comments above public functions that lack one.
    Asks the model only for comment text, does file manipulation in Python.
    """
    lines = file_path.read_text().splitlines()
    functions_needing_docs = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("pub fn ") or stripped.startswith("pub async fn "):
            j = i - 1
            while j >= 0 and lines[j].strip() == "":
                j -= 1
            if j < 0 or not lines[j].strip().startswith("///"):
                functions_needing_docs.append((i, line))

    if not functions_needing_docs:
        logger.info("[DOC] No public functions missing doc comments.")
        return False

    fn_list = "\n".join(
        f"Line {i+1}: {line.strip()}" for i, line in functions_needing_docs
    )
    prompt = f"""For each of the following Rust public functions, write a single concise /// doc comment (one line only).
Return a JSON array where each item has "line" (the line number) and "comment" (the full /// comment string).
Example: [{{"line": 42, "comment": "/// Initializes the vault and loads existing keys."}}]

Functions:
{fn_list}
"""
    response = router.complete(
        role="implementer",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )

    content = response.content.strip()
    if content.startswith("```"):
        content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))

    try:
        comments = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                comments = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("[DOC] Failed to parse doc comments from model response")
                return False
        else:
            logger.error("[DOC] Failed to parse doc comments from model response")
            return False

    comment_map = {item["line"]: item["comment"] for item in comments}

    for i, line in reversed(functions_needing_docs):
        line_num = i + 1
        comment = comment_map.get(line_num)
        if comment:
            indent = len(line) - len(line.lstrip())
            comment_text = comment.strip()
            if not comment_text.startswith("///"):
                comment_text = "/// " + comment_text
            comment_line = " " * indent + comment_text
            lines.insert(i, comment_line)

    file_path.write_text("\n".join(lines) + "\n")
    logger.info(f"[DOC] Inserted {len(comment_map)} doc comments into {file_path.name}")
    return True


# ---------------------------------------------------------------------------
# Controller (v2)
# ---------------------------------------------------------------------------

class Controller:
    """
    The GLITCHLAB brainstem (v2).

    Context Router architecture:
      - Agents receive only what they need via TaskState.to_agent_summary()
      - ScopeResolver computes file context from imports/AST, not planner guesses
      - Prelude is available as a tool, not injected globally
      - TaskState persists structured memory across agent steps

    Pipeline: Plan → Implement → Test → Debug Loop → Security → Release → PR
    """

    def __init__(
        self,
        repo_path: Path,
        config: GlitchLabConfig | None = None,
        allow_core: bool = False,
        auto_approve: bool = False,
        test_command: str | None = None,
    ):
        self.repo_path = repo_path.resolve()
        self.config = config or load_config(repo_path)
        self.allow_core = allow_core
        self.auto_approve = auto_approve
        self.test_command = test_command

        # Core components
        self.router = Router(self.config)
        self.boundary = BoundaryEnforcer(self.config.boundaries.protected_paths)

        # Agents
        self.planner = PlannerAgent(self.router)
        self.implementer = ImplementerAgent(self.router)
        self.debugger = DebuggerAgent(self.router)
        self.security = SecurityAgent(self.router)
        self.release = ReleaseAgent(self.router)
        self.archivist = ArchivistAgent(self.router)

        # Run state (reset per-task)
        self._state: TaskState | None = None
        self._workspace: Workspace | None = None
        self._scope: ScopeResolver | None = None
        self._repo_index: Any = None
        self._repo_index_context: str = ""

        # History tracking
        self._history = TaskHistory(self.repo_path)

        # Prelude — available as tool context, NOT global prefix
        self._prelude = PreludeContext(self.repo_path)

    def run(self, task: Task) -> dict[str, Any]:
        """Execute the full agent pipeline for a task."""

        # --- EXECUTION GUARD (Manual Patch) ---
        # Check for uncommitted changes in the main repo, ignoring .glitchlab/
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        ).stdout

        # Filter out changes that are ONLY inside .glitchlab/ (logs, tasks, etc.)
        dirty_files = [
            line for line in status.splitlines() 
            if not line[3:].startswith(".glitchlab/")
        ]

        if dirty_files:
            console.print("[red]🚫 Cannot run: Main repository has uncommitted changes:[/]")
            for f in dirty_files[:5]:
                console.print(f"  [dim]{f}[/]")
            raise DirtyRepoError("Clean your repository before running GLITCHLAB.")

        # Check if local branch is behind remote
        try:
            subprocess.run(["git", "fetch", "--quiet"], cwd=self.repo_path, timeout=10)
            behind = subprocess.run(
                ["git", "rev-list", "HEAD..@{u}", "--count"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            ).stdout.strip()
            if behind and behind.isdigit() and int(behind) > 0:
                console.print(f"[red]🚫 Cannot run: Local branch is behind remote by {behind} commits. Please pull changes.[/]")
                raise DirtyRepoError("Local branch is behind remote.")
        except Exception:
            pass
        # --------------------------------------

        # Initialize structured task state
        self._state = TaskState(
            task_id=task.task_id,
            objective=task.objective,
            mode=task.mode or "evolution",
            risk_level=task.risk_level,
        )

        result: dict[str, Any] = {
            "task_id": task.task_id,
            "status": "pending",
            "events": [],
        }

        console.print(Panel(
            f"[bold green]Task:[/] {task.objective[:120]}\n"
            f"[bold]ID:[/] {task.task_id}  |  [bold]Source:[/] {task.source}\n"
            f"[bold]Risk:[/] {task.risk_level}  |  [bold]Mode:[/] {task.mode.upper()}",
            title="⚡ GLITCHLAB v2",
            subtitle="Build Weird. Ship Clean.",
            border_style="bright_green",
        ))

        try:
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

            # ── 2. Plan ──
            plan = self._run_planner(task, ws_path, failure_context)
            if plan.get("parse_error"):
                result["status"] = "plan_failed"
                return result

            # Update TaskState with plan output
            self._state.plan_steps = [
                StepState(
                    step_number=s.get("step_number", 0),
                    description=s.get("description", ""),
                    files=s.get("files", []),
                    action=s.get("action", ""),
                )
                for s in plan.get("steps", [])
            ]
            self._state.files_in_scope = plan.get("files_likely_affected", [])
            self._state.estimated_complexity = plan.get("estimated_complexity", "medium")
            self._state.requires_core_change = plan.get("requires_core_change", False)
            self._state.mark_phase("plan")
            self._state.persist(ws_path)

            # ── 3. Boundary Validation (Plan-Level) ──
            try:
                violations = self.boundary.check_plan(plan, self.allow_core)
                if violations:
                    self._log_event("core_override", {"files": violations})
                    console.print(f"[yellow]⚠ Core override granted for: {violations}[/]")
            except BoundaryViolation as e:
                console.print(f"[red]🚫 {e}[/]")
                result["status"] = "boundary_violation"
                return result

            # ── 4. Governance Mode Routing ──
            is_maintenance = task.mode == "maintenance"
            is_evolution = task.mode == "evolution"

            if is_evolution:
                self.config.intervention.pause_before_pr = True

            # Strict doc-only detection
            objective_lower = task.objective.lower()
            is_doc_only = (
                is_maintenance
                and any(term in objective_lower for term in ["doc", "documentation", "///"])
                and task.risk_level == "low"
                and all(step.get("action") == "modify" for step in plan.get("steps", []))
                and any(f.endswith(".rs") for f in plan.get("files_likely_affected", []))
            )

            # ── 4A. Maintenance: Surgical Documentation Path ──
            if is_doc_only:
                console.print(
                    "\n[bold dim]📄 [MAINTENANCE MODE] "
                    "Surgical documentation update — implementer bypassed.[/]"
                )

                impl = {
                    "changes": [],
                    "tests_added": [],
                    "commit_message": f"docs: update documentation for {task.task_id}",
                    "summary": "Surgical documentation insertion.",
                }
                applied = []

                for f in plan.get("files_likely_affected", []):
                    fpath = ws_path / f
                    if fpath.exists():
                        inserted = insert_doc_comments(fpath, self.router)
                        if inserted:
                            applied.append(f"DOC {f}")

                for entry in applied:
                    console.print(f"  [cyan]{entry}[/]")

            # ── 4B. Standard Execution Path ──
            else:
                impl = self._run_implementer(task, plan, ws_path)

                if impl.get("parse_error"):
                    result["status"] = "implementation_failed"
                    return result

                # Update TaskState with implementation output
                self._state.files_modified = [
                    c.get("file", "") for c in impl.get("changes", [])
                    if c.get("action") in ("modify", "create")
                ]
                self._state.files_created = [
                    c.get("file", "") for c in impl.get("changes", [])
                    if c.get("action") == "create"
                ]
                self._state.tests_added = [t.get("file", "") for t in impl.get("tests_added", [])]
                self._state.commit_message = impl.get("commit_message", "")
                self._state.implementation_summary = impl.get("summary", "")
                self._state.mark_phase("implement")
                self._state.persist(ws_path)

                is_high_complexity = plan.get("estimated_complexity", "").lower() in ["high", "large"]
                if is_high_complexity:
                    console.print("  [dim]High complexity: allowing full-file rewrites.[/]")

                try:
                    applied = apply_changes(
                        ws_path,
                        impl.get("changes", []),
                        boundary=self.boundary,
                        allow_core=self.allow_core,
                        allow_test_modifications=not is_maintenance,
                        allow_full_rewrite=True,
                    )
                    applied += apply_tests(
                        ws_path,
                        impl.get("tests_added", []),
                        allow_test_modifications=not is_maintenance,
                    )
                except BoundaryViolation as e:
                    console.print(f"[red]🚫 Boundary Violation during implementation: {e}[/]")
                    result["status"] = "boundary_violation"
                    return result

                for entry in applied:
                    console.print(f"  [cyan]{entry}[/]")

            # ── Patch & Surgical failure retry (one attempt) ──
            # Catch BOTH "FAIL" (surgical) and "PATCH_FAILED" (diffs)
            patch_failures = [a for a in applied if "FAIL" in a or "PATCH_FAILED" in a]

            if patch_failures:
                console.print("[yellow]⚠ Edit failed to apply (likely a whitespace mismatch). Attempting one auto-repair...[/]")

                for entry in applied:
                    console.print(f"  [cyan]{entry}[/]")

                if any("FAIL" in a or "PATCH_FAILED" in a for a in applied):
                    console.print("[red]❌ Auto-repair failed. Aborting to prevent corrupted PR.[/]")
                    result["status"] = "implementation_failed"
                    return result

                retry_impl = self._retry_patch(task, plan, ws_path, impl, applied)

                if retry_impl.get("parse_error"):
                    result["status"] = "implementation_failed"
                    return result

                applied = apply_changes(
                    ws_path,
                    retry_impl.get("changes", []),
                    boundary=self.boundary,
                    allow_core=self.allow_core,
                    allow_test_modifications=not is_maintenance,
                    allow_full_rewrite=True,
                )

                for entry in applied:
                    console.print(f"  [cyan]{entry}[/]")

                if any(a.startswith("PATCH_FAILED") for a in applied):
                    console.print("[red]❌ Patch retry failed. Aborting.[/]")
                    result["status"] = "implementation_failed"
                    return result

            # ── Phase routing: doc-only skips test/security/release ──
            if is_doc_only:
                test_ok = True
                sec = {"verdict": "pass", "issues": []}
                rel = {
                    "version_bump": "none",
                    "reasoning": "Maintenance mode — documentation only",
                    "changelog_entry": "- Documentation updates",
                }
            else:
                # ── 5. Test + Debug Loop ──
                if self.test_command:
                    test_ok = self._run_fix_loop(task, ws_path, tools, impl)

                    if not test_ok:
                        result["status"] = "tests_failed"
                        console.print("[red]❌ Fix loop exhausted. Tests still failing.[/]")
                        if not self._confirm("Continue to PR anyway?"):
                            return result
                else:
                    test_ok = True

                self._state.test_passing = test_ok
                self._state.mark_phase("test")
                self._state.persist(ws_path)

                # ── 6. Security Review ──
                sec = self._run_security(task, impl, ws_path)

                self._state.security_verdict = sec.get("verdict", "")
                self._state.mark_phase("security")

                if sec.get("verdict") == "block":
                    console.print("[red]🚫 Security blocked this change.[/]")
                    self._print_security_issues(sec)

                    if self.auto_approve:
                        console.print("[red]❌ Auto-approve enabled. Aborting dangerous PR.[/]")
                        result["status"] = "security_blocked"
                        return result

                    if not self._confirm("Override security block?"):
                        result["status"] = "security_blocked"
                        return result

                # ── 7. Release Assessment ──
                rel = self._run_release(task, impl, ws_path)

                self._state.version_bump = rel.get("version_bump", "")
                self._state.changelog_entry = rel.get("changelog_entry", "")
                self._state.mark_phase("release")

                # ── 7.5. Archivist (Governed Documentation) ──
                nova_result = self._run_archivist(task, impl, plan, rel, ws_path)

                if is_maintenance:
                    nova_result["should_write_adr"] = False

                if nova_result.get("should_write_adr") and nova_result.get("adr"):
                    adr_applied = self._write_adr(ws_path, nova_result["adr"])
                    if adr_applied:
                        console.print(f"  [cyan]{adr_applied}[/]")

                for doc_update in nova_result.get("doc_updates", []):
                    doc_applied = self._write_doc_update(ws_path, doc_update)
                    if doc_applied:
                        console.print(f"  [cyan]{doc_applied}[/]")

                # Maintenance mode: forbid file create/delete and out-of-scope edits
                if is_maintenance:
                    allowed = set(plan.get("files_likely_affected") or [])
                    if not allowed:
                        raise RuntimeError(
                            "Maintenance mode requires explicit files_likely_affected"
                        )

                    for path in allowed:
                        self._workspace._git("add", path, check=False)

                    diff_output = self._workspace._git(
                        "diff", "--cached", "--name-status", check=False
                    )
                    lines = diff_output.splitlines() if diff_output else []

                    created, deleted, touched = [], [], []
                    for line in lines:
                        parts = line.split("\t", 1)
                        if len(parts) != 2:
                            continue
                        status, path = parts
                        if status == "A":
                            created.append(path)
                        elif status == "D":
                            deleted.append(path)
                        else:
                            touched.append(path)

                    out_of_scope = [p for p in touched if p not in allowed]
                    if created or deleted or out_of_scope:
                        raise RuntimeError(
                            f"Maintenance violation. "
                            f"created={created} deleted={deleted} "
                            f"out_of_scope={out_of_scope}"
                        )

            # ── 8. Commit + PR ──
            self._state.mark_phase("commit")
            self._state.persist(ws_path)

            commit_msg = impl.get("commit_message", f"glitchlab: {task.task_id}")
            self._workspace.commit(commit_msg)

            if self.config.intervention.pause_before_pr:
                diff = self._workspace.diff_stat()
                console.print(Panel(diff, title="Diff Summary", border_style="cyan"))
                if not self._confirm("Create PR?"):
                    result["status"] = "pr_cancelled"
                    return result

            try:
                self._workspace.push()
                pr_url = self._create_pr(task, impl, rel)
                result["pr_url"] = pr_url
                result["status"] = "pr_created"
                console.print(f"[bold green]✅ PR created: {pr_url}[/]")
            except Exception as e:
                logger.warning(f"PR creation failed: {e}")
                result["status"] = "committed"
                result["branch"] = self._workspace.branch_name
                console.print(f"[yellow]Changes committed to {self._workspace.branch_name}[/]")

            self._print_budget_summary()
            result["events"] = self._state.events
            result["budget"] = self.router.budget.summary()

            if getattr(task, "file_path", None) and task.file_path.exists() and task.file_path.parent.name == "queue":
                archive_dir = task.file_path.parent.with_name("archive")
                archive_dir.mkdir(parents=True, exist_ok=True)
                task.file_path.rename(archive_dir / task.file_path.name)
                console.print(f"[dim]Moved task file to {archive_dir / task.file_path.name}[/]")

        except BudgetExceededError as e:
            console.print(f"[red]💸 Budget exceeded: {e}[/]")
            result["status"] = "budget_exceeded"
        except KeyboardInterrupt:
            console.print("\n[yellow]⚡ Interrupted by human.[/]")
            result["status"] = "interrupted"
        except Exception as e:
            logger.exception("Controller error")
            console.print(f"[red]💥 Error: {e}[/]")
            result["status"] = "error"
            result["error"] = str(e)
        finally:
            if self._workspace:
                try:
                    self._workspace.cleanup()
                except Exception:
                    pass
            self._history.record(result)

        return result

    # -----------------------------------------------------------------------
    # Agent Runners — v2: Surgical context via TaskState + ScopeResolver
    # -----------------------------------------------------------------------

    def _run_planner(self, task: Task, ws_path: Path, failure_context: str = "") -> dict:
        console.print("\n[bold magenta]🧠 [ZAP] Planning...[/]")

        # Planner gets: repo file map + task + failure history
        # NO global Prelude dump. Prelude constraints already merged into task.
        objective_parts = []

        if self._repo_index_context:
            objective_parts.append(self._repo_index_context)

        if failure_context:
            objective_parts.append(failure_context)

        objective_parts.append(f"TASK:\n{task.objective}")

        objective = "\n\n---\n\n".join(objective_parts)

        context = AgentContext(
            task_id=task.task_id,
            objective=objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            constraints=task.constraints,
            acceptance_criteria=task.acceptance_criteria,
            risk_level=task.risk_level,
        )

        plan = self.planner.run(context)
        self._log_event("plan_created", {
            "steps": len(plan.get("steps", [])),
            "risk": plan.get("risk_level"),
        })

        self._print_plan(plan)

        if self.config.intervention.pause_after_plan and not self.auto_approve:
            if not self._confirm("Approve plan?"):
                plan["_aborted"] = True
                plan["parse_error"] = True
                return plan

        return plan

    def _run_implementer(self, task: Task, plan: dict, ws_path: Path) -> dict:
        console.print("\n[bold blue]🔧 [PATCH] Implementing...[/]")

        # v2: ScopeResolver computes context from actual imports
        # instead of blindly reading planner's guess list
        file_context = self._scope.resolve_for_files(
            plan.get("files_likely_affected", []),
            include_deps=True,
        )

        # Build constraints for reliable file operations
        impl_constraints = list(task.constraints)

        create_files = [
            s.get("files", []) for s in plan.get("steps", [])
            if s.get("action") == "create"
        ]
        if any(create_files):
            impl_constraints.append(
                "For NEW files (action='create'), always provide complete file "
                "content in the 'content' field. Never use unified diffs for new files."
            )

        impl_constraints.append(
            "For MODIFIED files (action='modify'), prefer providing complete file "
            "content in the 'content' field. Only use the 'patch' field if the file "
            "is large (>200 lines) and the change is small. Never wrap patches in "
            "markdown code fences."
        )

        # v2: Pass structured task state, not raw plan blob
        context = AgentContext(
            task_id=task.task_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            constraints=impl_constraints,
            acceptance_criteria=task.acceptance_criteria,
            file_context=file_context,
            previous_output=self._state.to_agent_summary("implementer"),
            extra={},
        )

        impl = self.implementer.run(context, max_tokens=12000)

        # For doc-comment tasks, use surgical insertion
        for change in impl.get("changes", []):
            if change.get("action") == "modify":
                fpath = ws_path / change["file"]
                if fpath.exists():
                    inserted = insert_doc_comments(fpath, self.router)
                    if inserted:
                        change["_doc_inserted"] = True
                        change["content"] = None

        self._log_event("implementation_created", {
            "changes": len(impl.get("changes", [])),
            "tests": len(impl.get("tests_added", [])),
        })

        return impl

    def _retry_patch(
        self,
        task: Task,
        plan: dict,
        ws_path: Path,
        original_impl: dict,
        applied_entries: list[str],
    ) -> dict:
        console.print("[dim]Re-prompting implementer with git error context...[/]")

        error_lines = [
            a.replace("PATCH_ERROR ", "")
            for a in applied_entries
            if a.startswith("PATCH_ERROR")
        ]
        git_error = "\n".join(error_lines)

        file_context = self._scope.resolve_for_files(
            plan.get("files_likely_affected", []),
            include_deps=False,
        )

        retry_prompt = f"""
The previous unified diff failed to apply with the following git error:

{git_error}

Regenerate a corrected unified diff that will apply cleanly.
Ensure:
- At least 3 context lines
- Exact whitespace match
- Valid unified diff format
- No unrelated changes
"""

        context = AgentContext(
            task_id=task.task_id,
            objective=task.objective + "\n\n" + retry_prompt,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            constraints=task.constraints,
            acceptance_criteria=task.acceptance_criteria,
            file_context=file_context,
            previous_output=self._state.to_agent_summary("implementer"),
        )

        return self.implementer.run(context, max_tokens=8192)

    def _run_fix_loop(
        self, task: Task, ws_path: Path, tools: ToolExecutor, impl: dict
    ) -> bool:
        """
        Run test → debug → fix loop (v2.1).
        
        Hardened to distinguish between:
          - Transport Failures: JSON truncation, parsing errors.
          - Logic Failures: Syntax errors, failing assertions.
        """
        max_attempts = self.config.limits.max_fix_attempts

        for attempt in range(1, max_attempts + 1):
            console.print(f"\n[bold]🧪 Test run {attempt}/{max_attempts}...[/]")

            try:
                # 1. Execute test command inside the isolated sandbox
                result = tools.execute(self.test_command)
            except ToolViolationError as e:
                console.print(f"[red]Tool violation: {e}[/]")
                return False

            if result.success:
                console.print("[green]✅ Tests pass![/]")
                self._log_event("tests_passed", {"attempt": attempt})
                return True

            error_output = result.stderr or result.stdout
            console.print(f"[red]❌ Tests failed (attempt {attempt})[/]")
            self._log_event("tests_failed", {"attempt": attempt})

            # 2. Update TaskState with error context
            self._state.last_error = (error_output or "")[:3000]
            self._state.debug_attempts = attempt

            if attempt >= max_attempts:
                break

            # 3. Invoke debugger with surgical context
            console.print(f"\n[bold yellow]🐛 [REROUTE] Debugging (attempt {attempt})...[/]")

            # Resolve file context for ONLY the files the implementer changed
            file_context = self._scope.resolve_for_files(
                [c["file"] for c in impl.get("changes", [])],
                include_deps=True,
            )

            # Detect prior failures to switch to "Full Content" strategy if patches fail
            patch_failed = any(
                "FAIL" in str(fix.get("_apply_result", ""))
                or "PATCH_FAILED" in str(fix.get("_apply_result", ""))
                for fix in self._state.previous_fixes
            )

            extra = {
                "error_output": (error_output or "")[:800], # Cap for JSON headroom
                "test_command": self.test_command,
                "attempt": attempt,
            }

            if patch_failed or attempt > 1:
                extra["patch_strategy"] = (
                    "IMPORTANT: Previous patches failed. Do NOT output unified diffs. "
                    "Provide the COMPLETE file content in the 'content' field."
                )

            context = AgentContext(
                task_id=task.task_id,
                objective=task.objective,
                repo_path=str(self.repo_path),
                working_dir=str(ws_path),
                file_context=file_context,
                previous_output=self._state.to_agent_summary("debugger"),
                extra=extra,
            )

            # 4. Invoke Reroute
            debug_result = self.debugger.run(context)
            
            # --- POLICY OVERRIDE: Transport vs Logic ---
            is_transport_failure = (
                debug_result.get("parse_error") or 
                debug_result.get("root_cause") == "JSON_TRUNCATION"
            )
            
            if is_transport_failure:
                console.print("[yellow]⚠ Transport failure (JSON truncation). Forcing retry...[/]")
                debug_result["should_retry"] = True  # System > Model Policy
            
            self._state.previous_fixes.append(debug_result)

            if not debug_result.get("should_retry", False): #
                console.print("[yellow]Debugger says: don't retry (logic failure).[/]")
                break

            # 5. Apply the fix
            fix_changes = debug_result.get("fix", {}).get("changes", [])
            if fix_changes:
                is_maintenance = (task.mode == "maintenance")
                fix_applied = apply_changes(
                    ws_path,
                    fix_changes,
                    boundary=self.boundary,
                    allow_core=self.allow_core,
                    allow_test_modifications=(not is_maintenance),
                    allow_full_rewrite=True,
                )
                for a in fix_applied:
                    console.print(f"  [cyan]{a}[/]")

                debug_result["_apply_result"] = fix_applied

                if all("FAIL" in a or "PATCH_FAILED" in a or "SKIP" in a for a in fix_applied):
                    console.print("[yellow]⚠ Debug fix failed to apply. Skipping re-test.[/]")
                    continue
            else:
                if not is_transport_failure:
                    console.print("[yellow]⚠ Debugger returned no fix changes.[/]")
                continue

        return False

    def _run_security(self, task: Task, impl: dict, ws_path: Path) -> dict:
        console.print("\n[bold red]🔒 [FRANKIE] Security scan...[/]")

        diff = self._workspace.diff_full() if self._workspace else ""

        # v2: Structured state, not raw impl blob
        context = AgentContext(
            task_id=task.task_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            previous_output=self._state.to_agent_summary("security"),
            extra={
                "diff": diff,
                "protected_paths": self.config.boundaries.protected_paths,
            },
        )

        result = self.security.run(context)
        self._log_event("security_review", {"verdict": result.get("verdict")})
        return result

    def _run_release(self, task: Task, impl: dict, ws_path: Path) -> dict:
        console.print("\n[bold cyan]📦 [SEMVER] Release assessment...[/]")

        diff = self._workspace.diff_stat() if self._workspace else ""

        context = AgentContext(
            task_id=task.task_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            previous_output=self._state.to_agent_summary("release"),
            extra={"diff": diff},
        )

        result = self.release.run(context)
        self._log_event("release_assessment", {"bump": result.get("version_bump")})
        return result

    def _run_archivist(
        self, task: Task, impl: dict, plan: dict, release: dict, ws_path: Path
    ) -> dict:
        """Run Archivist Nova with structured state context."""
        console.print("\n[bold dim]📚 [NOVA] Documenting...[/]")

        existing_docs = []
        for pattern in ["*.md", "docs/**/*.md", "doc/**/*.md"]:
            existing_docs.extend(
                str(p.relative_to(ws_path))
                for p in ws_path.glob(pattern)
                if p.is_file() and ".glitchlab" not in str(p)
            )

        # v2: Archivist gets structured summary, not raw blobs
        context = AgentContext(
            task_id=task.task_id,
            objective=task.objective,
            repo_path=str(self.repo_path),
            working_dir=str(ws_path),
            previous_output=self._state.to_agent_summary("archivist"),
            extra={
                "existing_docs": existing_docs[:20],
            },
        )

        result = self.archivist.run(context)
        self._log_event("archivist_completed", {
            "wrote_adr": result.get("should_write_adr", False),
            "doc_updates": len(result.get("doc_updates", [])),
        })
        return result

    @staticmethod
    def _write_adr(ws_path: Path, adr: dict) -> str | None:
        """Write an ADR to the workspace."""
        adr_dir = ws_path / ".context" / "decisions"
        if not adr_dir.exists():
            adr_dir = ws_path / "docs" / "adr"
        adr_dir.mkdir(parents=True, exist_ok=True)

        existing = list(adr_dir.glob("*.md"))
        next_num = len(existing) + 1

        title = adr.get("title", f"ADR-{next_num:03d}")
        safe_title = re.sub(r'[^a-z0-9\-]+', '-', title.lower())
        safe_title = re.sub(r'-+', '-', safe_title).strip('-')
        filename = f"{next_num:03d}-{safe_title[:50]}.md"
        filepath = adr_dir / filename

        content = f"""# {title}

**Status:** {adr.get('status', 'accepted')}
**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

## Context

{adr.get('context', 'N/A')}

## Decision

{adr.get('decision', 'N/A')}

## Consequences

{adr.get('consequences', 'N/A')}
"""
        alternatives = adr.get("alternatives_considered", [])
        if alternatives:
            content += "\n## Alternatives Considered\n\n"
            for alt in alternatives:
                content += f"- {alt}\n"

        content += "\n---\n*Generated by GLITCHLAB / Archivist Nova*\n"

        filepath.write_text(content)
        return f"ADR {filepath.relative_to(ws_path)}"

    @staticmethod
    def _write_doc_update(ws_path: Path, doc: dict) -> str | None:
        """Apply a documentation update."""
        fpath = ws_path / doc["file"]
        action = doc.get("action", "create")
        content = doc.get("content", "")

        if not content:
            return None

        if action == "create":
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)
            return f"DOC CREATE {doc['file']}"
        elif action == "append":
            fpath.parent.mkdir(parents=True, exist_ok=True)
            existing = fpath.read_text() if fpath.exists() else ""
            fpath.write_text(existing + "\n\n" + content)
            return f"DOC APPEND {doc['file']}"
        elif action == "update" and fpath.exists():
            fpath.write_text(content)
            return f"DOC UPDATE {doc['file']}"
        return None

    # -----------------------------------------------------------------------
    # PR Creation
    # -----------------------------------------------------------------------

    def _create_pr(self, task: Task, impl: dict, release: dict) -> str:
        """Create a GitHub PR via gh CLI."""
        title = impl.get("commit_message", f"glitchlab: {task.task_id}")
        body = self._build_pr_body(task, impl, release)

        result = subprocess.run(
            [
                "gh", "pr", "create",
                "--title", title,
                "--body", body,
                "--head", self._workspace.branch_name,
            ],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"PR creation failed: {result.stderr}")

        return result.stdout.strip()

    @staticmethod
    def _build_pr_body(task: Task, impl: dict, release: dict) -> str:
        body = f"""## 🔬 GLITCHLAB Automated PR

**Task:** {task.task_id}
**Source:** {task.source}

### Summary
{impl.get('summary', 'No summary provided.')}

### Changes
"""
        for change in impl.get("changes", []):
            body += f"- `{change.get('action', '?')}` {change.get('file', '?')}: {change.get('description', '')}\n"

        body += f"""
### Version Impact
- **Bump:** {release.get('version_bump', 'unknown')}
- **Reasoning:** {release.get('reasoning', 'N/A')}

### Changelog
{release.get('changelog_entry', 'N/A')}

---
*Generated by GLITCHLAB — Build Weird. Ship Clean.*
"""
        return body

    # -----------------------------------------------------------------------
    # Display Helpers
    # -----------------------------------------------------------------------

    def _print_plan(self, plan: dict) -> None:
        table = Table(title="Execution Plan", border_style="magenta")
        table.add_column("#", style="dim")
        table.add_column("Action")
        table.add_column("Files")
        table.add_column("Description")

        for step in plan.get("steps", []):
            table.add_row(
                str(step.get("step_number", "?")),
                step.get("action", "?"),
                ", ".join(step.get("files", [])),
                step.get("description", ""),
            )

        console.print(table)
        console.print(
            f"Risk: [bold]{plan.get('risk_level', '?')}[/] | "
            f"Core change: {plan.get('requires_core_change', False)} | "
            f"Complexity: {plan.get('estimated_complexity', '?')}"
        )

    def _print_security_issues(self, sec: dict) -> None:
        for issue in sec.get("issues", []):
            sev = issue.get("severity", "info")
            color = {"critical": "red", "high": "red", "medium": "yellow"}.get(sev, "dim")
            console.print(f"  [{color}]{sev.upper()}[/] {issue.get('description', '')}")

    def _print_budget_summary(self) -> None:
        summary = self.router.budget.summary()
        console.print(Panel(
            f"Tokens: {summary['total_tokens']:,} / "
            f"Cost: ${summary['estimated_cost']:.4f} / "
            f"Calls: {summary['call_count']}",
            title="💸 Budget",
            border_style="green",
        ))

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _confirm(self, prompt: str) -> bool:
        if self.auto_approve:
            return True
        return Confirm.ask(f"[bold]{prompt}[/]")

    def _log_event(self, event_type: str, data: dict | None = None) -> None:
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": self._state.task_id if self._state else None,
            "data": data or {},
        }
        if self._state:
            self._state.events.append(event)
        logger.debug(f"[EVENT] {event_type}: {data}")