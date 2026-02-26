"""
GLITCHLAB Auditor — Task Writer

Takes structured findings from the scanner and generates
well-scoped GLITCHLAB task YAML files using the OpenAI API.

One finding group → one task file.
Hard limit: max 3 files per task.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import ValidationError

from glitchlab.router import Router
from glitchlab.controller import Task  # Import the strict Pydantic schema
from .scanner import Finding, ScanResult


# ---------------------------------------------------------------------------
# Task Sizing
# ---------------------------------------------------------------------------

MAX_FILES_PER_TASK = 2
MAX_FINDINGS_PER_TASK = 5


def group_findings_into_tasks(result: ScanResult) -> list[list[Finding]]:
    """
    Group findings into task-sized chunks.

    Rules:
    - Max 3 files per task
    - Max 10 findings per task
    - Same kind of finding grouped together
    - High severity findings get their own task
    """
    tasks: list[list[Finding]] = []

    # Separate high severity findings — each gets its own task
    high_sev = [f for f in result.findings if f.severity == "high"]
    for finding in high_sev:
        tasks.append([finding])

    # Group remaining findings by kind, then by file
    remaining = [f for f in result.findings if f.severity != "high"]

    # Group by kind first
    by_kind: dict[str, list[Finding]] = {}
    for f in remaining:
        by_kind.setdefault(f.kind, []).append(f)

    for kind, findings in by_kind.items():
        # Further group by file batches of MAX_FILES_PER_TASK
        by_file: dict[str, list[Finding]] = {}
        for f in findings:
            by_file.setdefault(f.file, []).append(f)

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

    return tasks


# ---------------------------------------------------------------------------
# Task YAML Generator
# ---------------------------------------------------------------------------

class TaskWriter:
    """
    Generates GLITCHLAB task YAML files from scanner findings.
    Uses the router to call the model for task description generation
    and validates them against the Pydantic Task schema.
    """

    def __init__(self, router: Router, output_dir: Path):
        self.router = router
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_tasks(self, result: ScanResult) -> list[Path]:
        """Generate task YAML files for all findings. Returns list of written paths."""
        task_groups = group_findings_into_tasks(result)
        written = []

        logger.info(f"[AUDITOR] {len(result.findings)} findings → {len(task_groups)} tasks")

        for i, findings in enumerate(task_groups, 1):
            try:
                task_data = self._generate_task(findings, i)
                path = self._write_task_yaml(task_data, i)
                written.append(path)
                logger.info(f"[AUDITOR] Task {i}/{len(task_groups)}: {path.name}")
            except Exception as e:
                logger.error(f"[AUDITOR] Failed to generate task {i}: {e}")

        return written

    def _generate_task(self, findings: list[Finding], index: int) -> dict[str, Any]:
        """Ask the model to generate a well-scoped task, strictly validating as JSON."""
        findings_text = "\n".join(
            f"- [{f.kind}] {f.file}:{f.line} — {f.description}"
            for f in findings
        )

        files = list({f.file for f in findings})
        kind = findings[0].kind
        default_id = f"audit-{kind}-{index:03d}"

        prompt = f"""You are generating a GLITCHLAB task definition.

You MUST return ONLY a valid JSON object matching this exact schema:
{{
  "id": "{default_id}",
  "objective": "Clear, specific, actionable objective in one or two sentences",
  "constraints": ["constraint 1", "constraint 2"],
  "acceptance": ["criterion 1", "criterion 2"],
  "risk": "low"
}}

The following findings were detected in the codebase:
{findings_text}

Files affected: {', '.join(files)}
Finding type: {kind}

Rules for writing the task:
- The objective must be specific and actionable — tell the agent exactly what to do.
- Do not ask for more than what the findings show.
- Constraints must protect existing behavior (no logic changes for doc tasks, etc.).
- Acceptance criteria must be verifiable (e.g., "cargo test passes").
- Risk MUST BE exactly one of: "low", "medium", or "high". 
  Use "low" for doc/comment tasks, "medium" for refactors, "high" for core logic.
- Keep scope EXTREMELY tight — maximum {MAX_FILES_PER_TASK} files per task.
"""
        try:
            # Force the LLM to output a JSON object to eliminate YAML formatting hallucinations
            response = self.router.complete(
                role="planner",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                response_format={"type": "json_object"}
            )

            content = response.content.strip()

            # Strip markdown fences if the LLM leaked them despite json_object mode
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```") and not l.strip().lower() == "json"]
                content = "\n".join(lines)

            raw_data = json.loads(content)
            
            # Ensure the ID matches our required pattern if the LLM strayed
            if "id" not in raw_data or not raw_data["id"].startswith("audit-"):
                raw_data["id"] = default_id
                
            raw_data["source"] = "auditor"

            # STRICT VALIDATION: This passes the raw JSON through the Pydantic model.
            # If the LLM hallucinates an invalid risk level or misses a field, it throws a ValidationError.
            task_obj = Task(**raw_data)
            
            # Dump the validated object back to a dict using aliases (e.g. 'acceptance' instead of 'acceptance_criteria')
            task_data = task_obj.model_dump(by_alias=True, exclude_none=True)

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"[AUDITOR] JSON/Validation parse failed, using safe fallback: {e}")
            task_data = self._fallback_task(findings, index)
        except Exception as e:
            logger.warning(f"[AUDITOR] Unexpected error generating task, using fallback: {e}")
            task_data = self._fallback_task(findings, index)

        return task_data

    def _fallback_task(self, findings: list[Finding], index: int) -> dict[str, Any]:
        """Generate a safe fallback task if model output fails strict Pydantic validation."""
        files = list({f.file for f in findings})
        kind = findings[0].kind

        objectives = {
            "missing_doc": f"Add /// doc comments to public functions missing them in: {', '.join(files)}",
            "todo": f"Address TODO/FIXME comments in: {', '.join(files)}",
            "complex_function": f"Refactor complex functions in: {', '.join(files)}",
        }

        # This dictionary is guaranteed to pass `Task` validation
        return {
            "id": f"audit-{kind}-{index:03d}",
            "objective": objectives.get(kind, f"Fix {kind} issues in {', '.join(files)}"),
            "constraints": [
                "Do not modify function signatures",
                "Do not modify any logic unless explicitly required",
            ],
            "acceptance": ["cargo test passes", "Clean diff"],
            "risk": "low" if kind == "missing_doc" else "medium",
            "source": "auditor",
        }

    def _write_task_yaml(self, task_data: dict[str, Any], index: int) -> Path:
        """Write validated task data to a YAML file in the output directory."""
        task_id = task_data.get("id", f"audit-{index:03d}")
        
        # Sanitize filename to prevent path traversal or weird characters
        safe_id = re.sub(r"[^\w\-]", "-", task_id)
        path = self.output_dir / f"{safe_id}.yaml"

        # Write to disk as YAML for easy human review/editing
        with open(path, "w") as f:
            yaml.dump(
                task_data, 
                f, 
                default_flow_style=False, 
                allow_unicode=True, 
                sort_keys=False
            )

        return path