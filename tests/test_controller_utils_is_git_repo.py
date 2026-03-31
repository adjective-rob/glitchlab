from pathlib import Path

from glitchlab.controller_utils import is_git_repo


def test_is_git_repo_accepts_standard_repo_and_linked_worktree(tmp_path: Path) -> None:
    standard_repo = tmp_path / "standard"
    standard_repo.mkdir()
    (standard_repo / ".git").mkdir()

    linked_worktree = tmp_path / "worktree"
    linked_worktree.mkdir()
    (linked_worktree / ".git").write_text("gitdir: /tmp/real-git-dir\n")

    not_repo = tmp_path / "not_repo"
    not_repo.mkdir()

    assert is_git_repo(standard_repo) is True
    assert is_git_repo(linked_worktree) is True
    assert is_git_repo(not_repo) is False
