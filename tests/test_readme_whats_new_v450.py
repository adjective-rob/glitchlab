from pathlib import Path


def test_readme_whats_new_v450_mentions_version_sync_changes_only():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "## **What's New in v4.5.0**" in readme
    assert "### **Pipeline Reliability Hardening (v4.4.0–v4.5.0)**" in readme
    assert "### **Auditor & Governance**" in readme
    assert "### **Prelude Integration**" in readme
    assert "### **Previous: Zephyr SBOF Integration (v4.2.0)**" in readme

    assert "## **What's New in v4.3.0**" not in readme
