import json
from typing import Any

from glitchlab.agents import BaseAgent, AgentContext
from glitchlab.router import RouterResponse

class PerformanceAuditorAgent(BaseAgent):
    role = "performance_auditor"
    system_prompt = (
        "You are the Performance Auditor. Your job is to analyze code for I/O inefficiencies "
        "and resource leaks. Return your findings as a JSON object with a 'findings' list."
    )

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        content = f"Analyze the following files for performance issues:\n\n"
        for path, code in context.file_context.items():
            content += f"--- {path} ---\n{code}\n\n"
        return [self._system_msg(), self._user_msg(content)]

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"findings": [], "raw": response.content}
