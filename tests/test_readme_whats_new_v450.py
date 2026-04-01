from pathlib import Path


def test_readme_whats_new_v450_mentions_version_sync_changes_only():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "## **What's New in v4.5.0**" in readme
    assert "### **Version Synchronization Update**" in readme
    assert "GLITCHLAB v4.5.0 is a maintenance release focused on keeping the published version consistent everywhere it is surfaced." in readme
    assert "This release synchronizes the version value across the project so package metadata, public exports, and identity checks all report the same release:" in readme

    for bullet in [
        "* Updated package metadata in `pyproject.toml`",
        "* Updated the public version exports in `glitchlab/__init__.py`",
        "* Updated the public version export in `glitchlab/identity.py`",
        "* Updated the version assertion in `tests/test_identity.py`",
    ]:
        assert bullet in readme

    assert "### **Why It Matters**" in readme
    assert "Keeping these version declarations aligned reduces version drift and makes release reporting more reliable for tooling, imports, and tests." in readme

    assert "## **What's New in v4.3.0**" not in readme
