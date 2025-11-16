import shutil
from pathlib import Path

from dynamic_path_router import resolve_path, clear_cache  # noqa: E402


def test_resolve_path_external_location(tmp_path, monkeypatch):
    original = Path(resolve_path("sandbox_runner.py"))
    sandbox_file = original.name
    alt_root = tmp_path / "alt_root"
    alt_root.mkdir()
    shutil.copy2(original, alt_root / sandbox_file)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(alt_root))
    clear_cache()

    resolved = resolve_path(sandbox_file)
    assert resolved == alt_root / sandbox_file
