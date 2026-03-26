"""
GLITCHLAB Brain Writer — Persistent codebase memory.

After each successful run, extracts structural facts from the agent message
history and upserts them into:
  ~/.glitchlab/brain/codebase_heuristics.json

The implementer reads this at context-build time via run_implementer()
alongside the existing learned_heuristics from patterns.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


def upsert_brain(
    brain_dir: Path,
    repo_name: str,
    patterns: list[dict],
    impl_result: dict,
) -> None:
    """
    Extract facts from a successful run and upsert into the brain file.

    Args:
        brain_dir: Resolved path to the brain directory (from config.context.brain).
        repo_name: repo_path.name — used as the key namespace.
        patterns: Output of extract_patterns_from_messages() for this run.
        impl_result: The implementer's result dict (has 'changes', 'summary').
    """
    if not patterns:
        return

    brain_dir.mkdir(parents=True, exist_ok=True)
    brain_file = brain_dir / "codebase_heuristics.json"

    # Load existing brain or start fresh
    brain: dict[str, Any] = {}
    if brain_file.exists():
        try:
            brain = json.loads(brain_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[BRAIN] Could not read brain file, starting fresh: {e}")

    repo_brain: dict[str, Any] = brain.get(repo_name, {})

    for pattern in patterns:
        if pattern.get("outcome") != "pass":
            continue

        file_key = pattern.get("file_modified")
        if not file_key:
            continue

        entry: dict[str, Any] = repo_brain.get(file_key, {
            "always_read_alongside": [],
            "successful_edit_strategies": [],
            "common_failure_patterns": [],
            "run_count": 0,
        })

        # 1. always_read_alongside: accumulate files read before this write
        reads = pattern.get("files_read_first", [])
        for r in reads:
            if r != file_key and r not in entry["always_read_alongside"]:
                entry["always_read_alongside"].append(r)
        # Cap at 5 most relevant
        entry["always_read_alongside"] = entry["always_read_alongside"][:5]

        # 2. successful_edit_strategies: infer from tools used
        tools = pattern.get("tools_used", [])
        if "patch_function" in tools:
            strat = "patch_function preferred over replace_in_file"
        elif "write_file" in tools and "replace_in_file" not in tools:
            strat = "write_file (full rewrite) used successfully"
        elif "replace_in_file" in tools:
            strat = "replace_in_file (surgical edit) used successfully"
        else:
            strat = None

        if strat and strat not in entry["successful_edit_strategies"]:
            entry["successful_edit_strategies"].append(strat)
        entry["successful_edit_strategies"] = entry["successful_edit_strategies"][:3]

        entry["run_count"] = entry.get("run_count", 0) + 1
        repo_brain[file_key] = entry

    brain[repo_name] = repo_brain

    try:
        brain_file.write_text(json.dumps(brain, indent=2), encoding="utf-8")
        logger.debug(f"[BRAIN] Updated {len(repo_brain)} file entries for {repo_name}")
    except Exception as e:
        logger.warning(f"[BRAIN] Failed to write brain file: {e}")


def read_brain_hints(brain_dir: Path, repo_name: str, files_in_scope: list[str]) -> str:
    """
    Read brain hints for the given files and return a formatted string
    for injection into the implementer's user message.

    Returns empty string if no relevant hints exist.
    """
    brain_file = brain_dir / "codebase_heuristics.json"
    if not brain_file.exists():
        return ""

    try:
        brain = json.loads(brain_file.read_text(encoding="utf-8"))
    except Exception:
        return ""

    repo_brain = brain.get(repo_name, {})
    if not repo_brain:
        return ""

    parts = []
    for f in files_in_scope:
        entry = repo_brain.get(f)
        if not entry:
            continue
        lines = [f"- {f} (seen {entry.get('run_count', 0)} runs):"]
        alongside = entry.get("always_read_alongside", [])
        if alongside:
            lines.append(f"  Always read alongside: {', '.join(alongside)}")
        strategies = entry.get("successful_edit_strategies", [])
        if strategies:
            lines.append(f"  Proven strategies: {'; '.join(strategies)}")
        parts.append("\n".join(lines))

    if not parts:
        return ""

    return "Codebase memory (from prior runs):\n" + "\n".join(parts)
