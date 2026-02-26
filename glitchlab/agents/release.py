"""
📦 Semver Sam — Release + Version Guardian

Decides patch/minor/major.
Writes changelog entry.
Summarizes risk.

Energy: accountant with neon sneakers.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


class ReleaseAgent(BaseAgent):
    role = "release"

    system_prompt = """You are Semver Sam, the release guardian inside GLITCHLAB.

You analyze code changes and determine versioning impact.

You MUST respond with valid JSON only.

Output schema:
{
  "version_bump": "none|patch|minor|major",
  "reasoning": "Why this bump level",
  "changelog_entry": "Markdown changelog entry",
  "breaking_changes": [],
  "migration_notes": "Any migration needed, or null",
  "risk_summary": "Brief risk assessment for release"
}

Rules:
- patch: bug fixes, internal refactors, no API change
- minor: new features, non-breaking additions
- major: breaking changes to public API
- none: docs only, comments, formatting
- Be conservative. When in doubt, bump higher.
- Changelog should be clear and useful to humans.
"""

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        # v2: previous_output is TaskState.to_agent_summary("release")
        # Contains: task_id, objective, mode, risk_level,
        #           files_modified, implementation_summary, security_verdict
        state = context.previous_output
        diff_text = context.extra.get("diff", "No diff available")

        user_content = f"""Analyze these changes for version impact.

Task: {context.objective}
Task ID: {context.task_id}
Mode: {state.get('mode', 'evolution')}

Files modified: {state.get('files_modified', [])}
Implementation summary: {state.get('implementation_summary', 'No summary available')}
Security verdict: {state.get('security_verdict', 'not yet reviewed')}

Diff:
```
{diff_text[:5000]}
```

Determine version bump and write changelog entry as JSON."""

        return [self._system_msg(), self._user_msg(user_content)]

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        content = response.content.strip()

        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"[SEMVER] Failed to parse release JSON: {e}")
            result = {
                "version_bump": "patch",
                "reasoning": f"Could not parse: {e}",
                "changelog_entry": "- Changes applied (manual review needed)",
                "parse_error": True,
            }

        result["_agent"] = "release"
        result["_model"] = response.model
        result["_tokens"] = response.tokens_used
        result["_cost"] = response.cost

        logger.info(f"[SEMVER] Bump: {result.get('version_bump', '?')}")
        return result