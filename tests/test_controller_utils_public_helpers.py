from pathlib import Path
from types import SimpleNamespace
import subprocess

from glitchlab import controller_utils


def test_public_helpers_behavior_and_attestation_guard(monkeypatch, tmp_path):
    completed = subprocess.CompletedProcess(
        args=["git", "status"], returncode=0, stdout="ok\n", stderr=""
    )
    captured = {}

    def fake_run(cmd, cwd, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["timeout"] = timeout
        return completed

    monkeypatch.setattr(controller_utils.subprocess, "run", fake_run)

    result = controller_utils.run_git(["status"], cwd=tmp_path, timeout=7)
    assert result is completed
    assert captured == {
        "cmd": ["git", "status"],
        "cwd": tmp_path,
        "capture_output": True,
        "text": True,
        "timeout": 7,
    }

    assert controller_utils.calculate_quality_score({}, None) == {
        "score": 100,
        "tokens_used": 0,
        "debug_attempts": 0,
    }
    assert controller_utils.calculate_quality_score(
        {"total_tokens": 70000}, SimpleNamespace(debug_attempts=2)
    ) == {
        "score": 76,
        "tokens_used": 70000,
        "debug_attempts": 2,
    }

    emitted = []

    def fake_emit(**kwargs):
        emitted.append(kwargs)

    monkeypatch.setattr(controller_utils.bus, "emit", fake_emit)
    monkeypatch.setattr(controller_utils.uuid, "uuid4", lambda: "fixed-uuid")

    controller_utils.attest_controller_action("FAIL could not write", "run-1")
    controller_utils.attest_controller_action("contains ERROR marker", "run-1")
    assert emitted == []

    controller_utils.attest_controller_action("updated file successfully", "run-1")
    assert emitted == [
        {
            "event_type": "action.completed",
            "payload": {
                "command": "controller.write_file",
                "stdout": "updated file successfully",
                "stderr": "",
                "returncode": 0,
                "allowed": True,
            },
            "agent_id": "controller",
            "run_id": "run-1",
            "action_id": "act-fixed-uuid",
        }
    ]
