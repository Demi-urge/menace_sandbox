import importlib
import importlib.util
import pathlib
import socket
import sys

import pytest
from dynamic_path_router import resolve_path

spec = importlib.util.spec_from_file_location(
    "sandbox_runner.environment",
    resolve_path("sandbox_runner/environment.py"),
)
env = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
spec.loader.exec_module(env)


def test_patched_import_records_relative_path(tmp_path, monkeypatch):
    module_path = tmp_path / "foo.py"  # path-ignore
    module_path.write_text("x = 1\n")
    recorded: list[str] = []
    monkeypatch.setattr(env, "record_module_usage", recorded.append, raising=False)
    monkeypatch.setattr(env, "ROOT", tmp_path, raising=False)
    sys.path.insert(0, str(tmp_path))
    try:
        with env._patched_imports():
            importlib.import_module("foo")
        assert recorded == ["foo.py"]  # path-ignore
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("foo", None)


def test_network_isolation_blocks_socket(monkeypatch):
    orig_socket = socket.socket
    code = env._inject_failure_modes("", {"network"})
    try:
        exec(code, {})
        with pytest.raises(OSError):
            socket.socket().connect(("example.com", 80))
    finally:
        socket.socket = orig_socket
