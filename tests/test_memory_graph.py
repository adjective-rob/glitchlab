import json
from pathlib import Path

import pytest

from glitchlab.memory_graph import CausalEdge, CausalMemoryGraph, MemoryNode


def test_empty_graph_returns_no_results(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)
    assert graph.query_for_files(["foo.py"]) == []
    assert graph.query_failure_patterns("foo.py") == []
    assert graph.query_prerequisite_reads("foo.py") == []
    assert graph.build_agent_context(["foo.py"], "planner") == ""


def test_record_and_query_modification_failure(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    graph.record_modification_failure(
        task_id="task-1",
        file_modified="src/parser.py",
        error_type="SyntaxError",
        error_detail="Unexpected token on line 42",
    )

    edges = graph.query_for_files(["src/parser.py"])
    assert len(edges) == 1
    assert edges[0].relation == "triggered"
    assert edges[0].cause.file == "src/parser.py"
    assert edges[0].effect.kind == "test_failure"
    assert "SyntaxError" in edges[0].effect.node_id


def test_record_and_query_fix_resolution(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    graph.record_modification_failure(
        task_id="task-1",
        file_modified="src/parser.py",
        error_type="SyntaxError",
        error_detail="Unexpected token",
    )
    graph.record_fix_resolution(
        task_id="task-1",
        file_fixed="src/lexer.py",
        root_cause_file="src/parser.py",
        fix_description="Fixed token regex in lexer",
    )

    patterns = graph.query_failure_patterns("src/parser.py")
    assert len(patterns) >= 1

    # Should find the resolution
    resolved = [p for p in patterns if p["fix_description"] != "(unresolved)"]
    assert len(resolved) == 1
    assert resolved[0]["root_cause_file"] == "src/lexer.py"
    assert "token regex" in resolved[0]["fix_description"]


def test_record_discovery_chain(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    graph.record_discovery_chain(
        task_id="task-1",
        files_read=["src/types.py", "src/utils.py"],
        file_modified="src/handler.py",
        outcome="pass",
    )

    prereqs = graph.query_prerequisite_reads("src/handler.py")
    assert "src/types.py" in prereqs
    assert "src/utils.py" in prereqs


def test_discovery_chain_skips_self_reference(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    graph.record_discovery_chain(
        task_id="task-1",
        files_read=["src/handler.py", "src/types.py"],
        file_modified="src/handler.py",
        outcome="pass",
    )

    prereqs = graph.query_prerequisite_reads("src/handler.py")
    assert "src/handler.py" not in prereqs
    assert "src/types.py" in prereqs


def test_record_security_flag(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    graph.record_security_flag(
        task_id="task-1",
        file_flagged="src/auth.py",
        finding="Hardcoded secret detected",
        severity="high",
    )

    edges = graph.query_for_files(["src/auth.py"])
    assert len(edges) == 1
    assert edges[0].relation == "flagged"
    assert "Hardcoded secret" in edges[0].effect.detail


def test_persistence_across_instances(tmp_path: Path):
    # Write with one instance
    graph1 = CausalMemoryGraph(tmp_path)
    graph1.record_modification_failure(
        task_id="task-1",
        file_modified="src/main.py",
        error_type="ImportError",
        error_detail="No module named foo",
    )

    # Read with a new instance
    graph2 = CausalMemoryGraph(tmp_path)
    edges = graph2.query_for_files(["src/main.py"])
    assert len(edges) == 1
    assert edges[0].effect.detail == "No module named foo"


def test_corrupt_lines_are_skipped(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    # Manually write corrupt + valid data
    graph.log_dir.mkdir(parents=True, exist_ok=True)

    valid_edge = CausalEdge(
        task_id="task-1",
        cause=MemoryNode(node_id="a", kind="modification", file="x.py"),
        effect=MemoryNode(node_id="b", kind="test_failure", file="x.py"),
        relation="triggered",
    )

    with open(graph.graph_file, "w") as f:
        f.write("corrupt json line\n")
        f.write(valid_edge.model_dump_json() + "\n")
        f.write("{broken\n")

    edges = graph.query_for_files(["x.py"])
    assert len(edges) == 1


def test_rotation_keeps_bounded(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)
    graph.MAX_EDGES = 20  # lower for testing

    for i in range(30):
        graph.record_modification_failure(
            task_id=f"task-{i}",
            file_modified=f"file_{i}.py",
            error_type="Error",
            error_detail=f"Error {i}",
        )

    # After 30 inserts with max 20, rotation fires multiple times.
    # The graph should never exceed MAX_EDGES on disk.
    lines = graph.graph_file.read_text().strip().splitlines()
    assert len(lines) <= graph.MAX_EDGES

    # Most recent should be preserved
    last = json.loads(lines[-1])
    assert last["task_id"] == "task-29"


def test_build_agent_context_planner(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    # Add failure pattern
    graph.record_modification_failure(
        task_id="task-1",
        file_modified="src/api.py",
        error_type="TypeError",
        error_detail="Expected str got int",
    )
    graph.record_fix_resolution(
        task_id="task-1",
        file_fixed="src/models.py",
        root_cause_file="src/api.py",
        fix_description="Cast field to str in model",
    )

    context = graph.build_agent_context(["src/api.py"], "planner")
    assert "MEMORY GRAPH" in context
    assert "src/models.py" in context
    assert "Cast field to str" in context


def test_build_agent_context_implementer(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    graph.record_discovery_chain(
        task_id="task-1",
        files_read=["src/config.py"],
        file_modified="src/server.py",
        outcome="pass",
    )

    context = graph.build_agent_context(["src/server.py"], "implementer")
    assert "MEMORY GRAPH" in context
    assert "src/config.py" in context


def test_build_agent_context_security(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    graph.record_security_flag(
        task_id="task-1",
        file_flagged="src/auth.py",
        finding="SQL injection via user input",
        severity="critical",
    )

    context = graph.build_agent_context(["src/auth.py"], "security")
    assert "MEMORY GRAPH" in context
    assert "SQL injection" in context


def test_build_agent_context_debugger(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    graph.record_modification_failure(
        task_id="task-1",
        file_modified="src/parser.py",
        error_type="ValueError",
        error_detail="Invalid token",
    )
    graph.record_fix_resolution(
        task_id="task-1",
        file_fixed="src/lexer.py",
        root_cause_file="src/parser.py",
        fix_description="Fixed regex for token matching",
    )

    context = graph.build_agent_context(["src/parser.py"], "debugger")
    assert "MEMORY GRAPH" in context
    assert "src/lexer.py" in context


def test_prerequisite_reads_sorted_by_frequency(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    # Record types.py as prerequisite 3 times, utils.py once
    for i in range(3):
        graph.record_discovery_chain(
            task_id=f"task-{i}",
            files_read=["src/types.py"],
            file_modified="src/handler.py",
            outcome="pass",
        )
    graph.record_discovery_chain(
        task_id="task-3",
        files_read=["src/utils.py"],
        file_modified="src/handler.py",
        outcome="pass",
    )

    prereqs = graph.query_prerequisite_reads("src/handler.py")
    assert prereqs[0] == "src/types.py"  # most frequent first


def test_low_confidence_filtered_from_prereqs(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    # Failed outcome = 0.5 confidence, below default 0.6 threshold
    graph.record_discovery_chain(
        task_id="task-1",
        files_read=["src/bad_lead.py"],
        file_modified="src/target.py",
        outcome="fail",
    )

    prereqs = graph.query_prerequisite_reads("src/target.py")
    assert "src/bad_lead.py" not in prereqs


def test_get_stats(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    graph.record_modification_failure("t1", "a.py", "Err", "detail")
    graph.record_fix_resolution("t1", "b.py", "a.py", "fixed it")
    graph.record_security_flag("t1", "c.py", "XSS", "high")

    stats = graph.get_stats()
    assert stats["total_edges"] == 3
    assert stats["indexed_files"] >= 3
    assert stats["relations"]["triggered"] == 1
    assert stats["relations"]["resolved"] == 1
    assert stats["relations"]["flagged"] == 1


def test_deduplication_in_query(tmp_path: Path):
    graph = CausalMemoryGraph(tmp_path)

    # Same edge recorded twice (e.g., from two instances)
    graph.record_modification_failure("t1", "a.py", "Err", "detail")
    graph.record_modification_failure("t1", "a.py", "Err", "detail")

    edges = graph.query_for_files(["a.py"])
    # Should deduplicate by (cause.node_id, effect.node_id)
    assert len(edges) == 1
