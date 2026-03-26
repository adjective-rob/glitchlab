"""
Task Definition and Change Applicator.

Extracted from controller.py — contains the Task Pydantic model
and functions for applying implementation changes to the workspace.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from glitchlab.governance import BoundaryEnforcer, BoundaryViolation


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
            id=f"interactive-{ts}", # Changed from task_id to id
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
        change = _normalize_change(change)
        action = change.get("action")
        filename = change.get("file")
        surgical_blocks = change.get("surgical_blocks") or []
        full_content = change.get("content")

        if not action or not filename:
            raise ValueError(f"Invalid change payload: {change}")
        if action in {"create", "modify"} and not full_content and not surgical_blocks:
            raise ValueError(f"Invalid change payload: {change}")

        # ── SAFETY CHECK: Detect native agent modifications ──
        # If an agent used 'write_file' tool, skip manual application.
        if change.get("_already_applied"):
            applied.append(f"AGENT_APPLIED {filename}")
            logger.info(f"[APPLY] Skipping {filename} — already applied by agent tool.")
            continue

        # Existing logic for files that still need applying...
        if boundary:
            boundary.check([filename], allow_core)

        fpath = working_dir / filename

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
