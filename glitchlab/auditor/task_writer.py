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

from glitchlab.router import Router
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
    Uses the router to call the model for task description generation.
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
        """Ask the model to generate a well-scoped task YAML from findings."""
        findings_text = "\n".join(
            f"- [{f.kind}] {f.file}:{f.line} — {f.description}"
            for f in findings
        )

        files = list({f.file for f in findings})
        kind = findings[0].kind

        prompt = f"""You are generating a GLITCHLAB task YAML for a software project.

GLITCHLAB task YAML format:
```yaml
id: <short-kebab-case-id>
objective: "<clear, specific, actionable objective in one or two sentences>"
constraints:
  - "<constraint 1>"
  - "<constraint 2>"
acceptance:
  - "<acceptance criterion 1>"
  - "<acceptance criterion 2>"
risk: low|medium|high
```

The following findings were detected in the codebase:
{findings_text}

Files affected: {', '.join(files)}
Finding type: {kind}

Rules for writing the task:
- The objective must be specific and actionable — tell the agent exactly what to do
- Do not ask for more than what the findings show
- Constraints must protect existing behavior (no logic changes for doc tasks, etc.)
- Acceptance criteria must be verifiable (cargo test passes, etc.)
- Risk should be "low" for doc/comment tasks, "medium" for refactors, "high" for core logic
- Keep scope EXTREMELY tight — maximum 2 files per task to ensure patch determinism

Return ONLY the YAML, no explanation, no markdown fences.
"""
        response = self.router.complete(
            role="planner",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )

        content = response.content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = "\n".join(
                l for l in content.split("\n")
                if not l.strip().startswith("```")
            )

        try:
            task_data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            logger.warning(f"[AUDITOR] YAML parse failed, using fallback: {e}")
            task_data = self._fallback_task(findings, index)

        # Ensure required fields
        if not isinstance(task_data, dict):
            task_data = self._fallback_task(findings, index)

        task_data.setdefault("id", f"audit-{kind}-{index:03d}")
        task_data.setdefault("risk", "low")
        task_data.setdefault("constraints", [])
        task_data.setdefault("acceptance", ["cargo test passes"])

        return task_data

    def _fallback_task(self, findings: list[Finding], index: int) -> dict[str, Any]:
        """Generate a safe fallback task if model output fails to parse."""
        files = list({f.file for f in findings})
        kind = findings[0].kind
        symbols = [f.symbol for f in findings[:5]]

        objectives = {
            "missing_doc": f"Add /// doc comments to public functions missing them in: {', '.join(files)}",
            "todo": f"Address TODO/FIXME comments in: {', '.join(files)}",
            "complex_function": f"Refactor complex functions in: {', '.join(files)}",
        }

        return {
            "id": f"audit-{kind}-{index:03d}",
            "objective": objectives.get(kind, f"Fix {kind} issues in {', '.join(files)}"),
            "constraints": [
                "Do not modify function signatures",
                "Do not modify any logic unless explicitly required",
            ],
            "acceptance": ["cargo test passes"],
            "risk": "low" if kind == "missing_doc" else "medium",
        }

    def _write_task_yaml(self, task_data: dict[str, Any], index: int) -> Path:
        """Write task data to a YAML file in the output directory."""
        task_id = task_data.get("id", f"audit-{index:03d}")
        # Sanitize filename
        safe_id = re.sub(r"[^\w\-]", "-", task_id)
        path = self.output_dir / f"{safe_id}.yaml"

        with open(path, "w") as f:
            yaml.dump(task_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return path
