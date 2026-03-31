from pathlib import Path
from subprocess import CompletedProcess

from glitchlab import controller_utils


def test_pre_task_git_fetch_soft_fails_on_git_fetch_error(monkeypatch):
    monkeypatch.setattr(controller_utils, "is_git_repo", lambda path: True)

    calls = []

    def fake_run_git(args, cwd, timeout=20):
        calls.append((args, cwd, timeout))
        return CompletedProcess(args=["git", *args], returncode=1, stdout="", stderr="network unavailable")

    warnings = []
    monkeypatch.setattr(controller_utils, "run_git", fake_run_git)
    monkeypatch.setattr(controller_utils.logger, "warning", lambda message: warnings.append(message))

    repo = Path("/tmp/repo")
    controller_utils.pre_task_git_fetch(repo)

    assert calls == [(["fetch", "origin", "main"], repo, 20)]
    assert warnings == ["[GIT] Pre-task fetch failed (soft): network unavailable"]
