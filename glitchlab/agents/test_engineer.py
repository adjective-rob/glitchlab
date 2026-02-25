class TestEngineerAgent(BaseAgent):
    """A strict TDD enforcer agent that outputs pytest file plans in JSON."""

    role = "test_engineer"

    system_prompt = (
        "You are a strict Test-Driven Development (TDD) enforcer. "
        "Your outputs must be a JSON object containing a list of pytest files to be created or modified. "
        "You receive a technical plan from the AgentContext and must respond with a JSON payload describing the pytest files. "
        "Your response MUST be valid JSON and contain no information about internal implementation details. "
        "Write tests that check strict JSON responses, Pydantic validation, and HTTP status codes. Ignore how the code is implemented internally."
    )

    def handle_plan(self, agent_context):
        """Process the technical plan and return a JSON payload describing pytest files to touch."""
        plan = agent_context.get('technical_plan')
        # The agent framework is expected to return a JSON payload; this is a placeholder implementation.
        # In actual runtime, the LLM will generate the proper file list.
        return {
            "pytest_files": ["tests/test_example.py"]
        }
