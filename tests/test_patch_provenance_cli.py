import shutil
import subprocess
import sys
from pathlib import Path


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    scripts = repo / "scripts"
    scripts.mkdir()
    src = Path(__file__).resolve().parents[1] / "scripts" / "check_patch_provenance.py"
    shutil.copy(src, scripts / "check_patch_provenance.py")
    subprocess.run(["git", "add", "scripts/check_patch_provenance.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)
    return repo


def test_ci_missing_patch_id(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "self_coding_example.py").write_text("print('x')\n")
    subprocess.run(["git", "add", "self_coding_example.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "update"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    res = subprocess.run(
        [
            sys.executable,
            "scripts/check_patch_provenance.py",
            "--ci",
            "--commit",
            commit,
        ],
        cwd=repo,
    )
    assert res.returncode == 1


def test_hook_missing_patch_id(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "self_coding_example.py").write_text("print('x')\n")
    subprocess.run(["git", "add", "self_coding_example.py"], cwd=repo, check=True)
    msg = repo / "msg.txt"
    msg.write_text("no patch id")
    res = subprocess.run(
        [sys.executable, "scripts/check_patch_provenance.py", str(msg)], cwd=repo
    )
    assert res.returncode == 1
