import json
from pathlib import Path
import pytest

from glitchlab.history import TaskHistory

def test_jsonl_append_only_logic(tmp_path: Path):
    history = TaskHistory(tmp_path)
    
    # Record first entry
    history.record({"task_id": "task_1", "status": "success"})
    
    # Record second entry
    history.record({"task_id": "task_2", "status": "failed"})
    
    # Verify file exists and has two lines
    assert history.history_file.exists()
    lines = history.history_file.read_text().strip().splitlines()
    assert len(lines) == 2
    
    # Verify content
    entry1 = json.loads(lines[0])
    assert entry1["task_id"] == "task_1"
    assert entry1["status"] == "success"
    
    entry2 = json.loads(lines[1])
    assert entry2["task_id"] == "task_2"
    assert entry2["status"] == "failed"

def test_rotate_file_if_needed(tmp_path: Path):
    history = TaskHistory(tmp_path)
    test_file = tmp_path / "test_rotate.jsonl"
    
    # Create a file with 10 lines
    lines = [f'{{"line": {i}}}\n' for i in range(10)]
    test_file.write_text("".join(lines))
    
    # Rotate with max_lines=5
    history._rotate_file_if_needed(test_file, max_lines=5)
    
    # Should keep the most recent 80% of max_lines, which is 4 lines
    # Wait, the logic is:
    # if len(lines) > max_lines:
    #     f.writelines(lines[-int(max_lines * 0.8):])
    # So it keeps the last int(5 * 0.8) = 4 lines.
    
    new_lines = test_file.read_text().splitlines()
    assert len(new_lines) == 4
    assert json.loads(new_lines[0])["line"] == 6
    assert json.loads(new_lines[-1])["line"] == 9

def test_corrupt_json_lines_do_not_crash_reader(tmp_path: Path):
    history = TaskHistory(tmp_path)
    
    # Write some valid and corrupt lines
    history.log_dir.mkdir(parents=True, exist_ok=True)
    with open(history.history_file, "w") as f:
        f.write('{"task_id": "task_1", "status": "success"}\n')
        f.write('corrupt line 1\n')
        f.write('{"task_id": "task_2", "status": "failed"}\n')
        f.write('{"broken_json": \n')
    
    # Test get_all
    all_entries = history.get_all()
    assert len(all_entries) == 2
    assert all_entries[0]["task_id"] == "task_1"
    assert all_entries[1]["task_id"] == "task_2"
    
    # Test get_recent
    recent_entries = history.get_recent(10)
    assert len(recent_entries) == 2
    assert recent_entries[0]["task_id"] == "task_1"
    assert recent_entries[1]["task_id"] == "task_2"

def test_record_patterns_rotation(tmp_path: Path):
    history = TaskHistory(tmp_path)
    
    # Record 600 patterns to trigger rotation (max_lines=500)
    patterns = [{"file_modified": f"file_{i}.py"} for i in range(600)]
    history.record_patterns("task_1", patterns)
    
    patterns_file = history.log_dir / "patterns.jsonl"
    assert patterns_file.exists()
    
    lines = patterns_file.read_text().strip().splitlines()
    # Should keep int(500 * 0.8) = 400 lines
    assert len(lines) == 400
    
    # The last line should be the 599th pattern
    last_entry = json.loads(lines[-1])
    assert last_entry["file_modified"] == "file_599.py"

def test_build_heuristics_with_corrupt_lines(tmp_path: Path):
    history = TaskHistory(tmp_path)
    patterns_file = history.log_dir / "patterns.jsonl"
    history.log_dir.mkdir(parents=True, exist_ok=True)
    
    with open(patterns_file, "w") as f:
        f.write('{"type": "discovery_pattern", "outcome": "pass", "file_modified": "target.py", "files_read_first": ["read.py"]}\n')
        f.write('corrupt pattern line\n')
        f.write('{"type": "failure_resolution", "file_modified": "target.py", "error_type": "SyntaxError", "resolution": "Fixed syntax"}\n')
    
    heuristics = history.build_heuristics(["target.py"])
    
    assert "Known patterns from previous runs:" in heuristics
    assert "target.py: Usually requires reading read.py" in heuristics
    assert "Failure Contexts to Avoid:" in heuristics
    assert "Avoid modifying target.py without addressing: SyntaxError" in heuristics
