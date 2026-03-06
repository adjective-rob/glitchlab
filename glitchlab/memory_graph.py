"""
GLITCHLAB Agent Memory Graph — Causal Knowledge Across Runs

Upgrades the flat append-only history (patterns.jsonl) into a causal graph
where nodes are events (discoveries, errors, fixes) and edges encode
causality (e.g., "SyntaxError in parser.py was caused by a change in lexer.py
and was resolved by fixing the token regex").

When agents encounter a familiar situation, they can traverse the graph to
find what worked before — turning GLITCHLAB from memoryless to adaptive.

Storage: .glitchlab/logs/memory_graph.jsonl (append-only, one edge per line)
Index:   .glitchlab/logs/memory_index.json (rebuilt on load, file→edges lookup)

Design constraints:
  - No external dependencies (stdlib + pydantic only)
  - O(1) append, O(files_in_scope) query
  - Bounded growth via rotation (max 1000 edges)
  - Deterministic — no LLM calls in this module
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Graph Primitives
# ---------------------------------------------------------------------------

class MemoryNode(BaseModel):
    """A single event in the causal graph."""
    node_id: str
    kind: Literal[
        "discovery",     # agent read a file and learned something
        "modification",  # agent wrote/changed a file
        "test_failure",  # test run failed
        "test_pass",     # test run passed
        "fix",           # debugger applied a fix
        "security_flag", # security agent flagged something
    ]
    file: str = ""
    detail: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CausalEdge(BaseModel):
    """
    A directed edge encoding causality between two events.

    cause → effect with a label describing the relationship.
    The edge also records the task context so we can weight
    by recency and relevance.
    """
    task_id: str
    cause: MemoryNode
    effect: MemoryNode
    relation: Literal[
        "triggered",   # cause triggered the effect (e.g., modification → test_failure)
        "resolved",    # cause resolved the effect (e.g., fix → test_pass)
        "informed",    # cause informed the effect (e.g., discovery → fix)
        "flagged",     # security scan flagged something about a modification
    ]
    confidence: float = 1.0  # 0.0–1.0, decays over time
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# The Memory Graph
# ---------------------------------------------------------------------------

class CausalMemoryGraph:
    """
    Append-only causal graph with file-indexed queries.

    The graph is stored as JSONL (one CausalEdge per line) and indexed
    in memory by file path for O(1) lookups during agent runs.
    """

    MAX_EDGES = 1000
    DECAY_FACTOR = 0.95  # per-query decay for older edges

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path.resolve()
        self.log_dir = self.repo_path / ".glitchlab" / "logs"
        self.graph_file = self.log_dir / "memory_graph.jsonl"
        self.index_file = self.log_dir / "memory_index.json"

        # In-memory index: file_path → list[CausalEdge]
        self._file_index: dict[str, list[CausalEdge]] = {}
        self._edges: list[CausalEdge] = []
        self._loaded = False

    # --- Persistence ---

    def _ensure_loaded(self) -> None:
        """Lazy-load edges from disk on first access."""
        if self._loaded:
            return
        self._loaded = True

        if not self.graph_file.exists():
            return

        try:
            for line in self.graph_file.read_text(encoding="utf-8").strip().splitlines():
                try:
                    edge = CausalEdge.model_validate_json(line)
                    self._edges.append(edge)
                    self._index_edge(edge)
                except Exception:
                    continue  # skip corrupt lines
        except Exception as e:
            logger.warning(f"[MEMORY] Failed to load graph: {e}")

    def _index_edge(self, edge: CausalEdge) -> None:
        """Add an edge to the in-memory file index."""
        for file_path in {edge.cause.file, edge.effect.file}:
            if file_path:
                self._file_index.setdefault(file_path, []).append(edge)

    def _append_edge(self, edge: CausalEdge) -> None:
        """Persist a single edge to disk."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.graph_file, "a", encoding="utf-8") as f:
                f.write(edge.model_dump_json() + "\n")
        except Exception as e:
            logger.warning(f"[MEMORY] Failed to write edge: {e}")

        self._edges.append(edge)
        self._index_edge(edge)
        self._rotate_if_needed()

    def _rotate_if_needed(self) -> None:
        """Keep the graph bounded. Drop oldest edges when over limit."""
        if len(self._edges) <= self.MAX_EDGES:
            return

        keep = self._edges[-int(self.MAX_EDGES * 0.8):]
        self._edges = keep

        # Rebuild file index
        self._file_index.clear()
        for edge in keep:
            self._index_edge(edge)

        # Rewrite file
        try:
            with open(self.graph_file, "w", encoding="utf-8") as f:
                for edge in keep:
                    f.write(edge.model_dump_json() + "\n")
        except Exception as e:
            logger.warning(f"[MEMORY] Failed to rotate graph: {e}")

    # --- Recording (called by Controller) ---

    def record_modification_failure(
        self,
        task_id: str,
        file_modified: str,
        error_type: str,
        error_detail: str,
    ) -> None:
        """Record: a file modification triggered a test failure."""
        self._ensure_loaded()
        self._append_edge(CausalEdge(
            task_id=task_id,
            cause=MemoryNode(
                node_id=f"{task_id}:mod:{file_modified}",
                kind="modification",
                file=file_modified,
                detail=f"Modified during task {task_id}",
            ),
            effect=MemoryNode(
                node_id=f"{task_id}:fail:{error_type}",
                kind="test_failure",
                file=file_modified,
                detail=error_detail[:500],
            ),
            relation="triggered",
        ))

    def record_fix_resolution(
        self,
        task_id: str,
        file_fixed: str,
        root_cause_file: str,
        fix_description: str,
    ) -> None:
        """Record: a fix in one file resolved a failure linked to another."""
        self._ensure_loaded()
        self._append_edge(CausalEdge(
            task_id=task_id,
            cause=MemoryNode(
                node_id=f"{task_id}:fix:{file_fixed}",
                kind="fix",
                file=file_fixed,
                detail=fix_description[:500],
            ),
            effect=MemoryNode(
                node_id=f"{task_id}:pass:{root_cause_file}",
                kind="test_pass",
                file=root_cause_file,
                detail=f"Tests passed after fix to {file_fixed}",
            ),
            relation="resolved",
        ))

    def record_discovery_chain(
        self,
        task_id: str,
        files_read: list[str],
        file_modified: str,
        outcome: Literal["pass", "fail"],
    ) -> None:
        """Record: reading certain files informed a modification."""
        self._ensure_loaded()
        for src_file in files_read:
            if not src_file or src_file == file_modified:
                continue
            self._append_edge(CausalEdge(
                task_id=task_id,
                cause=MemoryNode(
                    node_id=f"{task_id}:read:{src_file}",
                    kind="discovery",
                    file=src_file,
                    detail=f"Read before modifying {file_modified}",
                ),
                effect=MemoryNode(
                    node_id=f"{task_id}:mod:{file_modified}",
                    kind="modification",
                    file=file_modified,
                    detail=f"Outcome: {outcome}",
                ),
                relation="informed",
                confidence=1.0 if outcome == "pass" else 0.5,
            ))

    def record_security_flag(
        self,
        task_id: str,
        file_flagged: str,
        finding: str,
        severity: str,
    ) -> None:
        """Record: security scan flagged a modification."""
        self._ensure_loaded()
        self._append_edge(CausalEdge(
            task_id=task_id,
            cause=MemoryNode(
                node_id=f"{task_id}:mod:{file_flagged}",
                kind="modification",
                file=file_flagged,
            ),
            effect=MemoryNode(
                node_id=f"{task_id}:sec:{file_flagged}",
                kind="security_flag",
                file=file_flagged,
                detail=f"[{severity}] {finding[:300]}",
            ),
            relation="flagged",
        ))

    # --- Querying (called by Agents via Controller) ---

    def query_for_files(
        self,
        files_in_scope: list[str],
        max_results: int = 15,
    ) -> list[CausalEdge]:
        """
        Return causal edges relevant to the given files, sorted by
        recency and confidence. This is the primary query for agents.
        """
        self._ensure_loaded()
        relevant: list[CausalEdge] = []

        for fpath in files_in_scope:
            edges = self._file_index.get(fpath, [])
            relevant.extend(edges)

        # Deduplicate by (cause.node_id, effect.node_id)
        seen: set[tuple[str, str]] = set()
        unique: list[CausalEdge] = []
        for edge in relevant:
            key = (edge.cause.node_id, edge.effect.node_id)
            if key not in seen:
                seen.add(key)
                unique.append(edge)

        # Sort by timestamp descending (most recent first), then confidence
        unique.sort(
            key=lambda e: (e.timestamp, e.confidence),
            reverse=True,
        )

        return unique[:max_results]

    def query_failure_patterns(
        self,
        file_path: str,
    ) -> list[dict[str, str]]:
        """
        For a specific file, find historical patterns:
        "When this file was modified, what broke and how was it fixed?"

        Returns a list of {error, root_cause_file, fix_description}.
        """
        self._ensure_loaded()
        edges = self._file_index.get(file_path, [])

        # Find modification → failure edges for this file
        failures: dict[str, dict[str, str]] = {}
        for edge in edges:
            if edge.relation == "triggered" and edge.cause.file == file_path:
                fail_id = edge.effect.node_id
                failures[fail_id] = {
                    "error": edge.effect.detail,
                    "task_id": edge.task_id,
                }

        # Now find fix → pass edges that resolved those failures
        patterns: list[dict[str, str]] = []
        for edge in edges:
            if edge.relation == "resolved" and edge.effect.file == file_path:
                patterns.append({
                    "error": f"Failure in {file_path}",
                    "root_cause_file": edge.cause.file,
                    "fix_description": edge.cause.detail,
                    "task_id": edge.task_id,
                })

        # Also include unresolved failures
        resolved_tasks = {p["task_id"] for p in patterns}
        for fail_id, info in failures.items():
            if info["task_id"] not in resolved_tasks:
                patterns.append({
                    "error": info["error"],
                    "root_cause_file": file_path,
                    "fix_description": "(unresolved)",
                    "task_id": info["task_id"],
                })

        return patterns[-10:]  # most recent 10

    def query_prerequisite_reads(
        self,
        file_to_modify: str,
        min_confidence: float = 0.6,
    ) -> list[str]:
        """
        "Before modifying this file, what files should I read first?"

        Returns files sorted by frequency of successful informed edges.
        """
        self._ensure_loaded()
        edges = self._file_index.get(file_to_modify, [])

        read_counts: dict[str, int] = {}
        for edge in edges:
            if (
                edge.relation == "informed"
                and edge.effect.file == file_to_modify
                and edge.confidence >= min_confidence
            ):
                src = edge.cause.file
                if src:
                    read_counts[src] = read_counts.get(src, 0) + 1

        # Sort by frequency descending
        return sorted(read_counts, key=read_counts.get, reverse=True)[:5]

    # --- Context Building (for injection into agent prompts) ---

    def build_agent_context(
        self,
        files_in_scope: list[str],
        for_agent: str,
    ) -> str:
        """
        Build a natural-language context block from the memory graph
        for injection into an agent's prompt.

        Different agents get different views:
        - planner: prerequisite reads + past failure patterns
        - debugger: failure → fix causal chains
        - implementer: prerequisite reads
        - security: past security flags
        """
        self._ensure_loaded()

        if not files_in_scope or not self._edges:
            return ""

        parts: list[str] = []

        if for_agent in ("planner", "implementer"):
            # What files to read before modifying
            for target in files_in_scope[:5]:
                prereqs = self.query_prerequisite_reads(target)
                if prereqs:
                    parts.append(
                        f"- Before modifying {target}, "
                        f"historically useful to read: {', '.join(prereqs)}"
                    )

        if for_agent in ("planner", "debugger"):
            # Past failure patterns
            for target in files_in_scope[:5]:
                patterns = self.query_failure_patterns(target)
                for p in patterns[:2]:
                    if p["fix_description"] != "(unresolved)":
                        parts.append(
                            f"- {target}: past error \"{p['error'][:80]}\" "
                            f"was fixed in {p['root_cause_file']}: "
                            f"{p['fix_description'][:100]}"
                        )

        if for_agent == "security":
            # Past security flags
            edges = self.query_for_files(files_in_scope)
            for edge in edges:
                if edge.relation == "flagged":
                    parts.append(
                        f"- {edge.effect.file}: previously flagged — "
                        f"{edge.effect.detail[:120]}"
                    )

        if not parts:
            return ""

        header = "=== MEMORY GRAPH (causal patterns from previous runs) ===\n"
        return header + "\n".join(parts[:15])

    def get_stats(self) -> dict[str, Any]:
        """Return summary stats about the graph."""
        self._ensure_loaded()
        relation_counts: dict[str, int] = {}
        file_count = len(self._file_index)
        for edge in self._edges:
            relation_counts[edge.relation] = relation_counts.get(edge.relation, 0) + 1

        return {
            "total_edges": len(self._edges),
            "indexed_files": file_count,
            "relations": relation_counts,
        }
