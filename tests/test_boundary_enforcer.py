"""Tests for BoundaryEnforcer from glitchlab/governance/__init__.py."""

import pytest

from glitchlab.governance import BoundaryEnforcer, BoundaryViolation


def test_check_no_protected_paths_returns_empty():
    """check with no protected paths returns empty violations."""
    enforcer = BoundaryEnforcer(protected_paths=[])
    violations = enforcer.check(["src/main.py", "lib/utils.rs"])
    assert violations == []


def test_check_protected_path_flags_matching_files():
    """check with a protected path flags matching files."""
    enforcer = BoundaryEnforcer(protected_paths=["src/core"])
    violations = enforcer.check(["src/core/mod.rs"], allow_core=True)
    assert "src/core/mod.rs" in violations


def test_check_raises_boundary_violation_when_allow_core_false():
    """check raises BoundaryViolation when allow_core=False and violations exist."""
    enforcer = BoundaryEnforcer(protected_paths=["src/core"])
    with pytest.raises(BoundaryViolation):
        enforcer.check(["src/core/engine.py"], allow_core=False)


def test_check_returns_violations_when_allow_core_true():
    """check returns violations list (no raise) when allow_core=True."""
    enforcer = BoundaryEnforcer(protected_paths=["src/core"])
    violations = enforcer.check(["src/core/engine.py"], allow_core=True)
    assert violations == ["src/core/engine.py"]


def test_check_plan_extracts_files_from_steps_and_files_likely_affected():
    """check_plan extracts files from plan steps AND files_likely_affected."""
    enforcer = BoundaryEnforcer(protected_paths=["src/core"])
    plan = {
        "files_likely_affected": ["src/core/main.rs"],
        "steps": [
            {"files": ["src/core/lib.rs"]},
            {"files": ["src/utils/helper.rs"]},
        ],
    }
    violations = enforcer.check_plan(plan, allow_core=True)
    assert "src/core/main.rs" in violations
    assert "src/core/lib.rs" in violations
    assert "src/utils/helper.rs" not in violations


def test_check_plan_deduplicates_file_paths():
    """check_plan deduplicates file paths before checking."""
    enforcer = BoundaryEnforcer(protected_paths=["src/core"])
    plan = {
        "files_likely_affected": ["src/core/mod.rs"],
        "steps": [
            {"files": ["src/core/mod.rs"]},
        ],
    }
    violations = enforcer.check_plan(plan, allow_core=True)
    # The file appears in both places but should only be in violations once
    assert violations.count("src/core/mod.rs") == 1


def test_prefix_matching_protects_nested_paths():
    """Prefix matching works — src/core protects src/core/mod.rs."""
    enforcer = BoundaryEnforcer(protected_paths=["src/core"])
    violations = enforcer.check(
        ["src/core/deep/nested/file.rs"], allow_core=True
    )
    assert "src/core/deep/nested/file.rs" in violations


def test_non_matching_paths_pass_cleanly():
    """Non-matching paths pass cleanly."""
    enforcer = BoundaryEnforcer(protected_paths=["src/core", "glitchlab/controller.py"])
    violations = enforcer.check(
        ["src/utils/helper.py", "tests/test_foo.py", "docs/readme.md"],
        allow_core=True,
    )
    assert violations == []
