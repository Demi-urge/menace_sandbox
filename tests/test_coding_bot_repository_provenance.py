import subprocess
from pathlib import Path

import pytest

import coding_bot_interface


@pytest.fixture(autouse=True)
def _reset_patch_service():
    coding_bot_interface._reset_patch_provenance_service()
    yield
    coding_bot_interface._reset_patch_provenance_service()


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "CI Bot"], cwd=repo, check=True)


def test_repository_provenance_resolution(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    package_root = repo / "menace_sandbox"
    package_root.mkdir()
    module_path = package_root / "example_bot.py"
    module_path.write_text("print('hello world')\n", encoding="utf-8")

    subprocess.run(["git", "add", "menace_sandbox/example_bot.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "feat: add example bot"], cwd=repo, check=True)

    commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
        .decode("utf-8")
        .strip()
    )

    class DummyService:
        def get(self, commit_hash: str):
            assert commit_hash == commit
            return {"patch_id": 712, "commit": commit}

    monkeypatch.setattr(
        coding_bot_interface,
        "_PATCH_PROVENANCE_SERVICE",
        DummyService(),
        raising=False,
    )
    monkeypatch.setattr(
        coding_bot_interface,
        "_unsigned_provenance_allowed",
        lambda: False,
    )

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))

    windows_style_path = (
        "C:\\shadow\\clone\\" + module_path.relative_to(repo).as_posix().replace("/", "\\")
    )

    decision = coding_bot_interface._resolve_provenance_decision(
        "ExampleBot",
        windows_style_path,
        [],
        (None, None),
    )

    assert decision.available is True
    assert decision.mode == "signed"
    assert decision.source == "repository"
    assert decision.patch_id == 712
    assert decision.commit == commit
