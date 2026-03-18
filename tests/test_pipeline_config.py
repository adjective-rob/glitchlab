"""Tests for the pipeline schema defined in config_loader.py."""

import pytest
from pydantic import ValidationError

from glitchlab.config_loader import GlitchLabConfig, PipelineStep, load_config


EXPECTED_STEP_NAMES = [
    "planner",
    "implementer",
    "debugger",
    "testgen",
    "security",
    "release",
    "archivist",
]


class TestDefaultPipelineConfig:
    """Default config loads with the correct pipeline steps."""

    def test_load_config_returns_pipeline_steps(self):
        config = load_config()
        assert len(config.pipeline) == 7

    def test_pipeline_step_names_match_expected_order(self):
        config = load_config()
        names = [step.name for step in config.pipeline]
        assert names == EXPECTED_STEP_NAMES

    def test_pipeline_step_agent_roles(self):
        config = load_config()
        roles = {step.name: step.agent_role for step in config.pipeline}
        assert roles["planner"] == "planner"
        assert roles["implementer"] == "implementer"
        assert roles["testgen"] == "testgen"
        assert roles["debugger"] == "debugger"
        assert roles["security"] == "security"
        assert roles["release"] == "release"
        assert roles["archivist"] == "archivist"

    def test_archivist_is_not_required(self):
        config = load_config()
        archivist = [s for s in config.pipeline if s.name == "archivist"][0]
        assert archivist.required is False

    def test_required_steps_are_required(self):
        config = load_config()
        for step in config.pipeline:
            if step.name != "archivist":
                assert step.required is True, f"{step.name} should be required"


class TestPipelineStepValidation:
    """PipelineStep validates required fields."""

    def test_minimal_valid_step(self):
        step = PipelineStep(name="test_step", agent_role="tester")
        assert step.name == "test_step"
        assert step.agent_role == "tester"
        assert step.required is True
        assert step.skip_if == []
        assert step.reads == []
        assert step.writes == []

    def test_missing_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            PipelineStep(agent_role="tester")  # type: ignore[call-arg]

    def test_missing_agent_role_raises_validation_error(self):
        with pytest.raises(ValidationError):
            PipelineStep(name="test_step")  # type: ignore[call-arg]

    def test_missing_both_required_fields_raises_validation_error(self):
        with pytest.raises(ValidationError):
            PipelineStep()  # type: ignore[call-arg]


class TestSkipIfConditions:
    """skip_if accepts known condition strings."""

    def test_skip_if_doc_only(self):
        step = PipelineStep(name="s", agent_role="a", skip_if=["doc_only"])
        assert step.skip_if == ["doc_only"]

    def test_skip_if_fast_mode(self):
        step = PipelineStep(name="s", agent_role="a", skip_if=["fast_mode"])
        assert step.skip_if == ["fast_mode"]

    def test_skip_if_no_test_command(self):
        step = PipelineStep(name="s", agent_role="a", skip_if=["no_test_command"])
        assert step.skip_if == ["no_test_command"]

    def test_skip_if_multiple_conditions(self):
        step = PipelineStep(
            name="s",
            agent_role="a",
            skip_if=["doc_only", "fast_mode", "no_test_command"],
        )
        assert step.skip_if == ["doc_only", "fast_mode", "no_test_command"]

    def test_default_pipeline_skip_if_values(self):
        config = load_config()
        steps = {s.name: s for s in config.pipeline}
        assert steps["testgen"].skip_if == ["doc_only"]
        assert steps["debugger"].skip_if == ["doc_only", "no_test_command"]
        assert steps["security"].skip_if == ["doc_only"]
        assert steps["release"].skip_if == ["doc_only"]
        assert steps["planner"].skip_if == []
        assert steps["implementer"].skip_if == []
        assert steps["archivist"].skip_if == []


class TestPipelineInGlitchLabConfig:
    """Pipeline field integrates correctly with GlitchLabConfig."""

    def test_empty_pipeline_default(self):
        config = GlitchLabConfig()
        assert config.pipeline == []

    def test_config_with_pipeline_steps(self):
        config = GlitchLabConfig(
            pipeline=[
                PipelineStep(name="step1", agent_role="role1"),
                PipelineStep(name="step2", agent_role="role2", required=False),
            ]
        )
        assert len(config.pipeline) == 2
        assert config.pipeline[0].name == "step1"
        assert config.pipeline[1].required is False
