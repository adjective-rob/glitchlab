"""Tests for apply_changes from glitchlab/controller.py."""

import pytest

from glitchlab.controller import apply_changes, _normalize_change
from glitchlab.governance import BoundaryEnforcer, BoundaryViolation


def test_surgical_blocks_all_match(tmp_path):
    """All surgical blocks match and apply correctly."""
    target = tmp_path / "app.py"
    target.write_text("def hello():\n    return 'hello'\n\ndef world():\n    return 'world'\n")

    changes = [
        {
            "file": "app.py",
            "action": "modify",
            "surgical_blocks": [
                {"search": "return 'hello'", "replace": "return 'hi'"},
                {"search": "return 'world'", "replace": "return 'earth'"},
            ],
        }
    ]

    result = apply_changes(tmp_path, changes)

    assert len(result) == 1
    assert "SURGICAL" in result[0]
    assert "2 blocks" in result[0]

    content = target.read_text()
    assert "return 'hi'" in content
    assert "return 'earth'" in content
    assert "return 'hello'" not in content
    assert "return 'world'" not in content


def test_surgical_block_partial_fail_falls_back_to_full_content(tmp_path):
    """One surgical block fails to match, falls back to full content."""
    target = tmp_path / "app.py"
    target.write_text("original content\n")

    fallback_content = "completely new content\n"
    changes = [
        {
            "file": "app.py",
            "action": "modify",
            "surgical_blocks": [
                {"search": "this does not exist", "replace": "replacement"},
            ],
            "content": fallback_content,
        }
    ]

    result = apply_changes(tmp_path, changes)

    assert len(result) == 1
    assert "MODIFY" in result[0]
    assert "full content" in result[0]
    assert target.read_text() == fallback_content


def test_full_content_write_on_existing_file(tmp_path):
    """Full content write on an existing file (action=modify)."""
    target = tmp_path / "config.yaml"
    target.write_text("old: value\n")

    new_content = "new: value\nupdated: true\n"
    changes = [
        {
            "file": "config.yaml",
            "action": "modify",
            "content": new_content,
        }
    ]

    result = apply_changes(tmp_path, changes)

    assert len(result) == 1
    assert "MODIFY" in result[0]
    assert target.read_text() == new_content


def test_file_creation_with_content(tmp_path):
    """File creation (action=create) with content."""
    changes = [
        {
            "file": "subdir/new_file.py",
            "action": "create",
            "content": "print('hello')\n",
        }
    ]

    result = apply_changes(tmp_path, changes)

    assert len(result) == 1
    assert "CREATE" in result[0]

    created = tmp_path / "subdir" / "new_file.py"
    assert created.exists()
    assert created.read_text() == "print('hello')\n"


def test_file_creation_missing_content_returns_fail(tmp_path):
    """File creation with missing content returns FAIL."""
    changes = [
        {
            "file": "empty.py",
            "action": "create",
        }
    ]

    result = apply_changes(tmp_path, changes)

    assert len(result) == 1
    assert "FAIL" in result[0]
    assert not (tmp_path / "empty.py").exists()


def test_file_deletion_removes_file(tmp_path):
    """File deletion (action=delete) removes the file."""
    target = tmp_path / "to_delete.txt"
    target.write_text("delete me\n")
    assert target.exists()

    changes = [
        {
            "file": "to_delete.txt",
            "action": "delete",
        }
    ]

    result = apply_changes(tmp_path, changes)

    assert len(result) == 1
    assert "DELETE" in result[0]
    assert not target.exists()


def test_boundary_violation_raises_when_allow_core_false(tmp_path):
    """Boundary violation raises BoundaryViolation when allow_core=False."""
    boundary = BoundaryEnforcer(protected_paths=["glitchlab/controller.py"])

    changes = [
        {
            "file": "glitchlab/controller.py",
            "action": "modify",
            "content": "hacked\n",
        }
    ]

    with pytest.raises(BoundaryViolation):
        apply_changes(tmp_path, changes, boundary=boundary, allow_core=False)


def test_already_applied_flag_skips_file(tmp_path):
    """The _already_applied flag causes the file to be skipped."""
    target = tmp_path / "touched.py"
    original = "untouched content\n"
    target.write_text(original)

    changes = [
        {
            "file": "touched.py",
            "action": "modify",
            "content": "should not be written\n",
            "_already_applied": True,
        }
    ]

    result = apply_changes(tmp_path, changes)

    assert len(result) == 1
    assert "AGENT_APPLIED" in result[0]
    assert target.read_text() == original


def test_allow_full_rewrite_false_blocks_fallback(tmp_path):
    """allow_full_rewrite=False blocks full content fallback in maintenance mode."""
    target = tmp_path / "main.py"
    target.write_text("original\n")

    changes = [
        {
            "file": "main.py",
            "action": "modify",
            "surgical_blocks": [
                {"search": "nonexistent text", "replace": "replacement"},
            ],
            "content": "full rewrite attempt\n",
        }
    ]

    result = apply_changes(tmp_path, changes, allow_full_rewrite=False)

    assert len(result) == 1
    assert "FAIL" in result[0]
    assert "full rewrite blocked" in result[0]
    assert target.read_text() == "original\n"


def test_normalize_change_promotes_patch_to_content():
    """_normalize_change promotes patch to content when patch is not a diff."""
    change = {
        "file": "example.py",
        "patch": "def hello():\n    return 'world'\n",
    }

    normalized = _normalize_change(change)

    assert normalized["content"] == "def hello():\n    return 'world'\n"
    assert normalized["patch"] is None
