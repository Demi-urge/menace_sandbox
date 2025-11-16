import os
import tempfile
import threading
from pathlib import Path

from filelock import FileLock


def _atomic_write(path: Path, data: str, *, lock: FileLock) -> None:
    """Minimal atomic write helper for tests using ``os.replace``."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fh:
        fh.write(data)
        tmp = Path(fh.name)
    os.replace(tmp, path)


def _apply_patch(path: Path, content: str) -> None:
    lock = FileLock(str(path) + ".lock")
    with lock:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if text and not text.endswith("\n"):
            text += "\n"
        text += content
        _atomic_write(path, text, lock=lock)


def test_parallel_patch_attempts(tmp_path):
    target = tmp_path / "module.py"
    target.write_text("start\n", encoding="utf-8")

    def worker(i: int) -> None:
        _apply_patch(target, f"patch{i}\n")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "start"
    assert sorted(lines[1:]) == [f"patch{i}" for i in range(5)]
