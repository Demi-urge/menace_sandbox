import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


def _load_environment_module():
    path = Path(__file__).resolve().parents[1] / "sandbox_runner" / "environment.py"
    if not hasattr(sys.modules.get("dynamic_path_router"), "repo_root"):
        sys.modules["dynamic_path_router"] = types.SimpleNamespace(
            resolve_path=lambda p: Path(p),
            resolve_dir=lambda p: Path(p),
            resolve_module_path=lambda p: Path(p),
            path_for_prompt=lambda p: Path(p).as_posix(),
            repo_root=lambda: Path("."),
            get_project_root=lambda: Path("."),
        )
    if "filelock" not in sys.modules:
        class _StubFileLock:
            def __init__(self, lock_file: str | None = None, *args, **kwargs) -> None:
                self.lock_file = lock_file or ""
                self.is_locked = False
                self._context = types.SimpleNamespace(timeout=None, lock_counter=0)

            def acquire(self, *args, **kwargs):
                self.is_locked = True
                return True

            def release(self, *args, **kwargs) -> None:
                self.is_locked = False

        class _StubTimeout(Exception):
            pass

        sys.modules["filelock"] = types.SimpleNamespace(
            FileLock=_StubFileLock,
            Timeout=_StubTimeout,
        )
    if "error_logger" not in sys.modules:
        class _StubErrorLogger:
            def __init__(self, *args, **kwargs) -> None:
                self.entries = []

            def record(self, *args, **kwargs) -> None:
                self.entries.append((args, kwargs))

        sys.modules["error_logger"] = types.SimpleNamespace(ErrorLogger=_StubErrorLogger)
    spec = importlib.util.spec_from_file_location("sandbox_runner.environment", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("failed to load sandbox_runner.environment")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


environment = _load_environment_module()


@pytest.mark.parametrize(
    ("attr_name", "filename", "reader", "expected"),
    [
        (
            "_ACTIVE_CONTAINERS_FILE",
            Path("active_containers.json"),
            environment._read_active_containers_unlocked,
            [],
        ),
        (
            "_ACTIVE_OVERLAYS_FILE",
            Path("active_overlays.json"),
            environment._read_active_overlays_unlocked,
            [],
        ),
        (
            "_FAILED_OVERLAYS_FILE",
            Path("failed_overlays.json"),
            environment._read_failed_overlays,
            [],
        ),
        (
            "FAILED_CLEANUP_FILE",
            Path("failed_cleanup.json"),
            environment._read_failed_cleanup,
            {},
        ),
        (
            "_CLEANUP_STATS_FILE",
            Path("cleanup_stats.json"),
            environment._read_cleanup_stats,
            {},
        ),
        (
            "_LAST_AUTOPURGE_FILE",
            Path("last_autopurge.json"),
            environment._read_last_autopurge,
            0.0,
        ),
    ],
)
def test_persistence_readers_ignore_empty_files(
    monkeypatch, tmp_path, caplog, attr_name, filename, reader, expected
) -> None:
    target = tmp_path / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()

    monkeypatch.setattr(environment, attr_name, target)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=environment.logger.name):
        result = reader()

    assert result == expected
    warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert not warnings
