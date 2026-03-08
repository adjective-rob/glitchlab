"""Tests for conditional agent activation rules in the Controller."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from glitchlab.controller import Controller, TaskState, StepState


# ---------------------------------------------------------------------------
# Helpers: build a minimal Controller with mocked dependencies
# ---------------------------------------------------------------------------


def _make_controller() -> Controller:
    """Create a Controller with enough internals for skip-logic testing."""
    ctrl = object.__new__(Controller)
    ctrl._state = TaskState(task_id="test-1", objective="fix a bug")
    ctrl._state.risk_level = "low"
    return ctrl


def _make_state(**overrides) -> TaskState:
    defaults = dict(task_id="t-1", objective="do something")
    defaults.update(overrides)
    return TaskState(**defaults)


# ===================================================================
# _should_skip_security
# ===================================================================


class TestSkipSecurity:
    def test_skip_when_low_risk_and_few_files(self):
        ctrl = _make_controller()
        ctrl._state.risk_level = "low"
        ctrl._state.files_modified = ["a.py", "b.py"]
        ctrl._state.files_created = []

        skip, reason = ctrl._should_skip_security({})
        assert skip is True
        assert "risk_level=low" in reason

    def test_no_skip_when_risk_is_medium(self):
        ctrl = _make_controller()
        ctrl._state.risk_level = "medium"
        ctrl._state.files_modified = ["a.py"]
        ctrl._state.files_created = []

        skip, _ = ctrl._should_skip_security({})
        assert skip is False

    def test_no_skip_when_risk_is_high(self):
        ctrl = _make_controller()
        ctrl._state.risk_level = "high"
        ctrl._state.files_modified = ["a.py"]
        ctrl._state.files_created = []

        skip, _ = ctrl._should_skip_security({})
        assert skip is False

    def test_no_skip_when_many_files_changed(self):
        ctrl = _make_controller()
        ctrl._state.risk_level = "low"
        ctrl._state.files_modified = [f"file_{i}.py" for i in range(5)]
        ctrl._state.files_created = []

        skip, _ = ctrl._should_skip_security({})
        assert skip is False

    def test_no_skip_when_low_risk_but_exactly_5_files(self):
        ctrl = _make_controller()
        ctrl._state.risk_level = "low"
        ctrl._state.files_modified = [f"f{i}.py" for i in range(5)]
        ctrl._state.files_created = []

        skip, _ = ctrl._should_skip_security({})
        assert skip is False

    def test_skip_counts_created_and_modified_together(self):
        ctrl = _make_controller()
        ctrl._state.risk_level = "low"
        ctrl._state.files_modified = ["a.py", "b.py"]
        ctrl._state.files_created = ["c.py", "d.py"]

        skip, _ = ctrl._should_skip_security({})
        assert skip is True  # 4 unique files, still < 5

    def test_no_skip_created_pushes_over_threshold(self):
        ctrl = _make_controller()
        ctrl._state.risk_level = "low"
        ctrl._state.files_modified = ["a.py", "b.py", "c.py"]
        ctrl._state.files_created = ["d.py", "e.py"]

        skip, _ = ctrl._should_skip_security({})
        assert skip is False  # 5 unique files


# ===================================================================
# _should_skip_release
# ===================================================================


class TestSkipRelease:
    def test_skip_when_no_version_files_touched(self):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["src/utils.py", "src/models.py"]
        ctrl._state.files_created = []

        skip, reason = ctrl._should_skip_release()
        assert skip is True
        assert "no version-bearing" in reason

    def test_no_skip_when_pyproject_modified(self):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["pyproject.toml"]
        ctrl._state.files_created = []

        skip, _ = ctrl._should_skip_release()
        assert skip is False

    def test_no_skip_when_package_json_modified(self):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["package.json"]
        ctrl._state.files_created = []

        skip, _ = ctrl._should_skip_release()
        assert skip is False

    def test_no_skip_when_cargo_toml_modified(self):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["Cargo.toml"]
        ctrl._state.files_created = []

        skip, _ = ctrl._should_skip_release()
        assert skip is False

    def test_no_skip_when_version_py_in_path(self):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["src/version.py"]
        ctrl._state.files_created = []

        skip, _ = ctrl._should_skip_release()
        assert skip is False

    def test_no_skip_when_setup_cfg_created(self):
        ctrl = _make_controller()
        ctrl._state.files_modified = []
        ctrl._state.files_created = ["setup.cfg"]

        skip, _ = ctrl._should_skip_release()
        assert skip is False

    def test_skip_when_only_source_files(self):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["lib/handler.rs", "lib/router.rs"]
        ctrl._state.files_created = ["lib/new_module.rs"]

        skip, _ = ctrl._should_skip_release()
        assert skip is True


# ===================================================================
# _should_skip_archivist
# ===================================================================


class TestSkipArchivist:
    def test_skip_when_no_docs_and_no_arch_changes(self):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["src/utils.py"]
        ctrl._state.files_created = []
        ctrl._state.requires_core_change = False
        ctrl._state.plan_steps = [
            StepState(step_number=1, description="Fix off-by-one error")
        ]

        skip, reason = ctrl._should_skip_archivist({})
        assert skip is True
        assert "no documentation" in reason

    def test_no_skip_when_md_file_modified(self):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["README.md"]
        ctrl._state.files_created = []
        ctrl._state.requires_core_change = False
        ctrl._state.plan_steps = []

        skip, _ = ctrl._should_skip_archivist({})
        assert skip is False

    def test_no_skip_when_docs_dir_file_modified(self):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["docs/api-guide.txt"]
        ctrl._state.files_created = []
        ctrl._state.requires_core_change = False
        ctrl._state.plan_steps = []

        skip, _ = ctrl._should_skip_archivist({})
        assert skip is False

    def test_no_skip_when_objective_mentions_architecture(self):
        ctrl = _make_controller()
        ctrl._state.objective = "Refactor the authentication architecture"
        ctrl._state.files_modified = ["src/auth.py"]
        ctrl._state.files_created = []
        ctrl._state.requires_core_change = False
        ctrl._state.plan_steps = []

        skip, _ = ctrl._should_skip_archivist({})
        assert skip is False

    def test_no_skip_when_plan_step_mentions_migration(self):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["src/db.py"]
        ctrl._state.files_created = []
        ctrl._state.requires_core_change = False
        ctrl._state.plan_steps = [
            StepState(step_number=1, description="Add migration for new schema")
        ]

        skip, _ = ctrl._should_skip_archivist({})
        assert skip is False

    def test_no_skip_when_requires_core_change(self):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["src/core.py"]
        ctrl._state.files_created = []
        ctrl._state.requires_core_change = True
        ctrl._state.plan_steps = []

        skip, _ = ctrl._should_skip_archivist({})
        assert skip is False

    def test_no_skip_when_rst_docs_created(self):
        ctrl = _make_controller()
        ctrl._state.files_modified = []
        ctrl._state.files_created = ["docs/changelog.rst"]
        ctrl._state.requires_core_change = False
        ctrl._state.plan_steps = []

        skip, _ = ctrl._should_skip_archivist({})
        assert skip is False

    def test_skip_objective_has_no_arch_keywords(self):
        ctrl = _make_controller()
        ctrl._state.objective = "fix typo in error message"
        ctrl._state.files_modified = ["src/errors.py"]
        ctrl._state.files_created = []
        ctrl._state.requires_core_change = False
        ctrl._state.plan_steps = [
            StepState(step_number=1, description="Change string literal")
        ]

        skip, _ = ctrl._should_skip_archivist({})
        assert skip is True


# ===================================================================
# _should_skip_testgen
# ===================================================================


class TestSkipTestgen:
    def test_skip_when_tests_exist(self, tmp_path):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["src/utils.py"]
        ctrl._state.files_created = []

        # Create the corresponding test file
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_utils.py").write_text("def test_it(): pass")

        skip, reason = ctrl._should_skip_testgen(tmp_path)
        assert skip is True
        assert "tests already exist" in reason

    def test_no_skip_when_no_tests_exist(self, tmp_path):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["src/utils.py"]
        ctrl._state.files_created = []

        skip, _ = ctrl._should_skip_testgen(tmp_path)
        assert skip is False

    def test_skip_when_only_test_files_modified(self, tmp_path):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["tests/test_foo.py"]
        ctrl._state.files_created = []

        skip, reason = ctrl._should_skip_testgen(tmp_path)
        assert skip is True
        assert "no non-test source files" in reason

    def test_no_skip_when_mixed_files_and_one_lacks_tests(self, tmp_path):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["src/utils.py", "src/handler.py"]
        ctrl._state.files_created = []

        # Only create test for utils, not handler
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_utils.py").write_text("def test_it(): pass")

        skip, _ = ctrl._should_skip_testgen(tmp_path)
        assert skip is False

    def test_skip_when_sibling_test_file_exists(self, tmp_path):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["src/utils.py"]
        ctrl._state.files_created = []

        # Create test as sibling: src/test_utils.py
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "test_utils.py").write_text("def test_it(): pass")

        skip, _ = ctrl._should_skip_testgen(tmp_path)
        assert skip is True

    def test_skip_when_all_files_have_tests(self, tmp_path):
        ctrl = _make_controller()
        ctrl._state.files_modified = ["src/a.py", "src/b.py"]
        ctrl._state.files_created = []

        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_a.py").write_text("pass")
        (tmp_path / "tests" / "test_b.py").write_text("pass")

        skip, _ = ctrl._should_skip_testgen(tmp_path)
        assert skip is True
