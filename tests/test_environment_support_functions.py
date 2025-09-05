import importlib.util
import sys
import types
from pathlib import Path


# Copied helper for importing environment.py with minimal dependencies  # path-ignore
def _load_env():
    if "filelock" not in sys.modules:
        class DummyLock:
            def __init__(self):
                self.is_locked = False
                self.lock_file = ""

            def acquire(self):
                self.is_locked = True

            def release(self):
                self.is_locked = False

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, exc_type, exc, tb):
                self.release()

        sys.modules["filelock"] = types.SimpleNamespace(FileLock=lambda *a, **k: DummyLock())

    pkg_path = Path(__file__).resolve().parents[1] / "sandbox_runner"
    if "sandbox_runner" not in sys.modules:
        pkg = types.ModuleType("sandbox_runner")
        pkg.__path__ = [str(pkg_path)]
        sys.modules["sandbox_runner"] = pkg
    path = pkg_path / "environment.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("sandbox_runner.environment", path)
    env = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner.environment"] = env
    assert spec and spec.loader
    spec.loader.exec_module(env)  # type: ignore[attr-defined]
    return env


def test_rlimits_supported_missing_resource(monkeypatch):
    env = _load_env()
    monkeypatch.setattr(env, "resource", None, raising=False)
    assert env._rlimits_supported() is False


def test_rlimits_supported_present(monkeypatch):
    env = _load_env()
    res_mod = types.SimpleNamespace(
        RLIMIT_CPU=0,
        RLIMIT_AS=9,
        getrlimit=lambda limit: (0, 0),
    )
    monkeypatch.setattr(env, "resource", res_mod, raising=False)
    assert env._rlimits_supported() is True


def test_psutil_rlimits_supported_missing(monkeypatch):
    env = _load_env()
    monkeypatch.setattr(env, "psutil", None, raising=False)
    assert env._psutil_rlimits_supported() is False


def test_psutil_rlimits_supported_present(monkeypatch):
    env = _load_env()

    class DummyProc:
        def rlimit(self, *a, **k):
            return None

    dummy_ps = types.SimpleNamespace(Process=lambda: DummyProc())
    monkeypatch.setattr(env, "psutil", dummy_ps, raising=False)
    assert env._psutil_rlimits_supported() is True


def test_cgroup_v2_supported(tmp_path, monkeypatch):
    env = _load_env()
    monkeypatch.setattr(env, "_CGROUP_BASE", tmp_path, raising=False)
    assert env._cgroup_v2_supported() is False
    (tmp_path / "cgroup.controllers").write_text("cpu memory")
    assert env._cgroup_v2_supported() is True
