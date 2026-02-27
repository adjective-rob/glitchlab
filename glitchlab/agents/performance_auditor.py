import json
from typing import Any

from glitchlab.agents.base import BaseAgent

class PerformanceAuditorAgent(BaseAgent):
    role = "performance_auditor"
    
    # Core Model: Gemini 3 Flash
    # Note: Image generation (Nano Banana) and Video (Veo) are available tools
    
    system_prompt = (
        "You are the Performance Auditor. Your job is to analyze code for I/O inefficiencies."
    )

    def build_messages(self, context: Any) -> list[dict[str, str]]:
        # Logic here
        return [self._system_msg(), self._user_msg("Analyze this code...")]

    def parse_response(self, response: Any, context: Any) -> dict[str, Any]:
        return json.loads(response.content)