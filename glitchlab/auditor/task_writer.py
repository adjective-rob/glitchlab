"""
GLITCHLAB Auditor — Task Writer (v2: Scout Edition)

Takes structured findings from the scanner and generates
well-scoped, prioritized GLITCHLAB task YAML files.

v2 Upgrades:
  - Priority scoring: tasks sorted for optimal overnight batch runs
  - Category-aware grouping: security fixes first, then bugs, features last
  - Dependency-aware ordering: foundational tasks before dependent ones
  - Effort estimation: small/medium/large reflected in task metadata
  - Scout brain findings get richer task descriptions via LLM
  - Task queue manifest: summary YAML for human review before batch run
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import ValidationError

from glitchlab.router import Router
from glitchlab.controller import Task
from .scanner import Finding, ScanResult


# ---------------------------------------------------------------------------
# Task Sizing & Priority
# ---------------------------------------------------------------------------

MAX_FILES_PER_TASK = 3
MAX_FINDINGS_PER_TASK = 5

# Priority weights — lower number = runs first in batch queue
CATEGORY_PRIORITY = {
    "security": 0,    # Security fixes are always urgent
    "bug": 1,         # Bugs before features
    "test": 2,        # Tests before new code
    "refactor": 3,    # Clean up before building more
    "cleanup": 4,     # Housekeeping
    "docs": 5,        # Documentation
    "feature": 6,     # Features last — build on a clean foundation
}

SEVERITY_WEIGHT = {
    "high": 0,
    "medium": 10,
    "low": 20,
}

EFFORT_LABELS = {
    "small": "~15 min",
    "medium": "~45 min",
    "large": "~2 hours",
}


def compute_priority(findings: list[Finding]) -> int:
    """
    Compute a priority score for a group of findings.
    Lower = higher priority = runs first in batch.
    """
    if not findings:
        return 999

    # Use the most severe/urgent finding in the group
    best_category = min(CATEGORY_PRIORITY.get(f.category, 5) for f in findings)
    best_severity = min(SEVERITY_WEIGHT.get(f.severity, 20) for f in findings)

    return best_category * 100 + best_severity


def group_findings_into_tasks(result: ScanResult) -> list[list[Finding]]:
    """
    Group findings into task-sized chunks optimized for batch execution.

    Strategy:
    1. High severity findings → own task (immediate attention)
    2. Security findings → grouped by file, own tasks
    3. Remaining grouped by (category, kind) → then by file batches
    4. Each group respects MAX_FILES_PER_TASK and MAX_FINDINGS_PER_TASK
    """
    tasks: list[list[Finding]] = []

    # Phase 1: High severity findings get their own task
    high_sev = [f for f in result.findings if f.severity == "high"]
    for finding in high_sev:
        tasks.append([finding])

    # Phase 2: Group remaining by (category, kind) for coherent tasks
    remaining = [f for f in result.findings if f.severity != "high"]

    by_category_kind: dict[tuple[str, str], list[Finding]] = {}
    for f in remaining:
        key = (f.category, f.kind)
        by_category_kind.setdefault(key, []).append(f)

    for (category, kind), findings in by_category_kind.items():
        # Sub-group by file for tight scope
        by_file: dict[str, list[Finding]] = {}
        for f in findings:
            by_file.setdefault(f.file, []).append(f)

        # Batch files into groups of MAX_FILES_PER_TASK
        file_groups: list[list[str]] = []
        current_group: list[str] = []
        for file_path in by_file:
            current_group.append(file_path)
            if len(current_group) >= MAX_FILES_PER_TASK:
                file_groups.append(current_group)
                current_group = []
        if current_group:
            file_groups.append(current_group)

        for file_group in file_groups:
            group_findings = []
            for fp in file_group:
                group_findings.extend(by_file[fp])
            # Chunk by MAX_FINDINGS_PER_TASK
            for i in range(0, len(group_findings), MAX_FINDINGS_PER_TASK):
                tasks.append(group_findings[i:i + MAX_FINDINGS_PER_TASK])

    # Sort task groups by priority (security first, features last)
    tasks.sort(key=compute_priority)

    return tasks


# ---------------------------------------------------------------------------
# Task YAML Generator
# ---------------------------------------------------------------------------

class TaskWriter:
    """
    Generates prioritized, batch-ready GLITCHLAB task YAML files.

    Scout Edition features:
    - Priority-ordered task generation
    - Category-aware task descriptions
    - Effort estimation in task metadata
    - Queue manifest for human review
    """

    def __init__(self, router: Router, output_dir: Path):
        self.router = router
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_tasks(self, result: ScanResult) -> list[Path]:
        """Generate prioritized task YAML files. Returns list of written paths."""
        task_groups = group_findings_into_tasks(result)
        written: list[Path] = []
        task_manifest: list[dict[str, Any]] = []

        logger.info(f"[SCOUT] {len(result.findings)} findings → {len(task_groups)} tasks")

        for i, findings in enumerate(task_groups, 1):
            try:
                task_data = self._generate_task(findings, i, len(task_groups))
                path = self._write_task_yaml(task_data, i)
                written.append(path)

                # Build manifest entry
                task_manifest.append({
                    "file": path.name,
                    "priority": i,
                    "category": findings[0].category,
                    "severity": max((f.severity for f in findings), key=lambda s: {"high": 2, "medium": 1, "low": 0}.get(s, 0)),
                    "effort": max((f.effort for f in findings), key=lambda e: {"large": 2, "medium": 1, "small": 0}.get(e, 0)),
                    "objective": task_data.get("objective", "")[:100],
                    "files": list({f.file for f in findings}),
                })

                logger.info(f"[SCOUT] Task {i}/{len(task_groups)}: {path.name} [{findings[0].category}]")
            except Exception as e:
                logger.error(f"[SCOUT] Failed to generate task {i}: {e}")

        # Write the queue manifest for human review
        if task_manifest:
            self._write_manifest(task_manifest, result)

        return written

    def _generate_task(self, findings: list[Finding], index: int, total: int) -> dict[str, Any]:
        """Generate a well-scoped task with priority and effort metadata."""
        findings_text = "\n".join(
            f"- [{f.severity.upper()}] [{f.category}] {f.file}:{f.line} — {f.description}"
            for f in findings
        )

        files = list({f.file for f in findings})
        kind = findings[0].kind
        category = findings[0].category
        effort = max((f.effort for f in findings), key=lambda e: {"large": 2, "medium": 1, "small": 0}.get(e, 0))
        priority = compute_priority(findings)
        default_id = f"scout-{category}-{kind}-{index:03d}"

        # Category-specific generation instructions
        category_instructions = {
            "security": "Focus on fixing the vulnerability. Acceptance criteria MUST include 'No known CVEs in dependencies' or 'Vulnerability patched'. Risk should be 'high'.",
            "bug": "Describe the exact bug pattern and how to fix it. Include a test that would catch the regression. Risk should be 'medium' or 'high'.",
            "test": "Generate test cases that cover the identified gaps. Acceptance criteria should include specific test scenarios. Risk should be 'low'.",
            "refactor": "Describe the refactor clearly. Emphasize that behavior must not change. Include 'All existing tests pass' in acceptance. Risk should be 'medium'.",
            "cleanup": "Keep scope tight. Only touch what's listed. Risk should be 'low'.",
            "docs": "Add documentation without changing any logic. Risk should be 'low'.",
            "feature": "Describe the feature clearly with specific acceptance criteria. Consider edge cases. Risk should be 'medium'.",
        }

        extra_instructions = category_instructions.get(category, "")

        prompt = f"""You are Scout, generating a GLITCHLAB task definition.

You MUST return ONLY a valid JSON object matching this exact schema:
{{
  "id": "{default_id}",
  "objective": "Clear, specific, actionable objective in one or two sentences",
  "constraints": ["constraint 1", "constraint 2"],
  "acceptance": ["criterion 1", "criterion 2"],
  "risk": "low"
}}

## Findings (Priority {index}/{total})
{findings_text}

Files affected: {', '.join(files)}
Category: {category}
Estimated effort: {effort} ({EFFORT_LABELS.get(effort, '?')})

## Category-Specific Instructions
{extra_instructions}

## Rules
- The objective must be specific and actionable — tell the agent exactly what to do.
- Do not ask for more than what the findings show.
- Constraints must protect existing behavior (no logic changes for doc tasks, etc.).
- Acceptance criteria must be verifiable (e.g., "cargo test passes", "no bare except blocks remain").
- Risk MUST BE exactly one of: "low", "medium", or "high".
- Keep scope EXTREMELY tight — maximum {MAX_FILES_PER_TASK} files per task.
- For security findings, always set risk to "high".
- For bug findings, always set risk to at least "medium".
- The task should be completable by an autonomous agent in a single session.
"""
        try:
            response = self.router.complete(
                role="scout",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                response_format={"type": "json_object"}
            )

            content = response.content.strip()

            # Strip markdown fences
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [ln for ln in lines if not ln.strip().startswith("```") and not ln.strip().lower() == "json"]
                content = "\n".join(lines)

            raw_data = json.loads(content)

            if "id" not in raw_data or not raw_data["id"].startswith("scout-"):
                raw_data["id"] = default_id

            raw_data["source"] = "auditor"

            # Strict validation through Pydantic Task model
            task_obj = Task(**raw_data)
            task_data = task_obj.model_dump(by_alias=True, exclude_none=True)

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"[SCOUT] Validation failed, using fallback: {e}")
            task_data = self._fallback_task(findings, index)
        except Exception as e:
            logger.warning(f"[SCOUT] Unexpected error, using fallback: {e}")
            task_data = self._fallback_task(findings, index)

        # Enrich with Scout metadata (not part of Task schema, but useful for queue ordering)
        task_data["_scout"] = {
            "priority": priority,
            "category": category,
            "effort": effort,
            "effort_estimate": EFFORT_LABELS.get(effort, "unknown"),
            "finding_count": len(findings),
            "files": files,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        return task_data

    def _fallback_task(self, findings: list[Finding], index: int) -> dict[str, Any]:
        """Generate a safe fallback task if model output fails validation."""
        files = list({f.file for f in findings})
        kind = findings[0].kind
        category = findings[0].category

        objectives = {
            "missing_doc": f"Add doc comments to public functions missing them in: {', '.join(files)}",
            "todo": f"Address TODO/FIXME comments in: {', '.join(files)}",
            "complex_function": f"Refactor complex functions (>60 lines) in: {', '.join(files)}",
            "dead_code": f"Remove dead code and unused imports in: {', '.join(files)}",
            "error_handling": f"Fix error handling gaps (bare except, unwrap chains) in: {', '.join(files)}",
            "code_smell": f"Clean up code smells (deep nesting, magic numbers) in: {', '.join(files)}",
            "test_gap": f"Add test coverage for untested modules: {', '.join(files)}",
            "dependency_vuln": f"Update vulnerable dependencies flagged in: {', '.join(files)}",
            "feature_opportunity": f"Implement identified feature improvement in: {', '.join(files)}",
            "bug_risk": f"Fix identified bug risk in: {', '.join(files)}",
            "performance": f"Address performance issue in: {', '.join(files)}",
            "api_inconsistency": f"Fix API inconsistencies in: {', '.join(files)}",
        }

        risk_map = {
            "security": "high",
            "bug": "medium",
            "feature": "medium",
            "refactor": "medium",
            "test": "low",
            "cleanup": "low",
            "docs": "low",
        }

        return {
            "id": f"scout-{category}-{kind}-{index:03d}",
            "objective": objectives.get(kind, f"Fix {kind} issues in {', '.join(files)}"),
            "constraints": [
                "Do not modify function signatures unless required by the fix",
                "Do not modify unrelated logic",
            ],
            "acceptance": [
                "All existing tests pass",
                "Clean diff — only touches relevant code",
            ],
            "risk": risk_map.get(category, "low"),
            "source": "auditor",
        }

    def _write_task_yaml(self, task_data: dict[str, Any], index: int) -> Path:
        """Write validated task data to a YAML file."""
        task_id = task_data.get("id", f"scout-{index:03d}")

        safe_id = re.sub(r"[^\w\-]", "-", task_id)
        path = self.output_dir / f"{safe_id}.yaml"

        # Separate Scout metadata from the task schema
        scout_meta = task_data.pop("_scout", None)

        # Write task data (schema-valid) with Scout metadata as YAML comment header
        with open(path, "w") as f:
            if scout_meta:
                f.write(f"# Scout Priority: {scout_meta.get('priority', '?')}\n")
                f.write(f"# Category: {scout_meta.get('category', '?')}\n")
                f.write(f"# Effort: {scout_meta.get('effort', '?')} ({scout_meta.get('effort_estimate', '?')})\n")
                f.write(f"# Files: {', '.join(scout_meta.get('files', []))}\n")
                f.write(f"# Generated: {scout_meta.get('generated_at', '?')}\n")
                f.write("# ---\n")

            yaml.dump(
                task_data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )

        return path

    def _write_manifest(self, manifest: list[dict[str, Any]], result: ScanResult) -> Path:
        """
        Write a queue manifest — a summary of all generated tasks for human review.

        This is the "mission briefing" the team reads before kicking off batch mode.
        """
        summary = result.summary()
        manifest_path = self.output_dir / "_scout_manifest.yaml"

        manifest_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scan_summary": {
                "files_scanned": summary["files_scanned"],
                "total_findings": summary["total"],
                "by_category": summary.get("by_category", {}),
                "by_severity": summary.get("by_severity", {}),
                "dependencies_checked": summary.get("dependency_count", 0),
            },
            "task_count": len(manifest),
            "execution_order": "Tasks are ordered by priority (security > bugs > tests > refactor > cleanup > docs > features)",
            "estimated_batch_time": self._estimate_batch_time(manifest),
            "tasks": manifest,
        }

        with open(manifest_path, "w") as f:
            f.write("# ============================================================\n")
            f.write("# SCOUT MISSION BRIEFING — Queue Manifest\n")
            f.write("# ============================================================\n")
            f.write("# Review this file before running batch mode.\n")
            f.write("# Remove any task entries you don't want executed.\n")
            f.write(f"# Run with: glitchlab batch --repo <path> --tasks-dir {self.output_dir}\n")
            f.write("# ============================================================\n\n")
            yaml.dump(
                manifest_data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info(f"[SCOUT] Manifest written: {manifest_path}")
        return manifest_path

    def _estimate_batch_time(self, manifest: list[dict]) -> str:
        """Rough estimate of how long a batch run will take."""
        effort_minutes = {"small": 15, "medium": 45, "large": 120}
        total_minutes = sum(
            effort_minutes.get(t.get("effort", "medium"), 45)
            for t in manifest
        )
        hours = total_minutes // 60
        mins = total_minutes % 60
        return f"~{hours}h {mins}m (sequential) / ~{max(1, hours // 3)}h {mins}m (3 workers)"
