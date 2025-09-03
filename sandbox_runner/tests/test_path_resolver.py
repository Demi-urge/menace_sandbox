import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from dynamic_path_router import resolve_path, clear_cache  # noqa: E402


def test_resolve_path_external_location(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    original = repo_root / "sandbox_runner.py"
    alt_root = tmp_path / "alt_root"
    alt_root.mkdir()
    shutil.copy2(original, alt_root / "sandbox_runner.py")

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(alt_root))
    clear_cache()

    resolved = resolve_path("sandbox_runner.py")
    assert resolved == alt_root / "sandbox_runner.py"
