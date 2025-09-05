import importlib.util
import os
import sys
import types
from pathlib import Path

import shutil

import pytest

from dynamic_path_router import resolve_dir, resolve_path


@pytest.fixture(scope="module")
def WorkflowSandboxRunner():
    package_path = resolve_dir("sandbox_runner")
    package = types.ModuleType("sandbox_runner")
    package.__path__ = [str(package_path)]
    sys.modules["sandbox_runner"] = package

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner.workflow_sandbox_runner",
        resolve_path("workflow_sandbox_runner.py"),  # path-ignore
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.WorkflowSandboxRunner


@pytest.fixture()
def runner(WorkflowSandboxRunner):
    return WorkflowSandboxRunner()


def test_files_confined_to_temp_dir(tmp_path, runner):
    outside = tmp_path / "outside.txt"

    def workflow():
        with open(outside, "w") as fh:
            fh.write("content")

    runner.run(workflow)
    assert not outside.exists()


def test_safe_mode_blocks_network_and_files(monkeypatch, tmp_path, runner):
    outside = tmp_path / "leak.txt"
    called: list[str] = []

    def fake_urlopen(url, *a, **kw):
        called.append(url)
        return b"ok"

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    def workflow():
        with open(outside, "w") as fh:
            fh.write("data")
        import urllib.request

        urllib.request.urlopen("http://example.com")

    metrics = runner.run(workflow, safe_mode=True)

    assert not outside.exists()
    assert called == []  # our monkeypatch was bypassed by safe_mode patching
    mod = metrics.modules[0]
    assert mod.success is False
    assert mod.exception and "file write disabled" in mod.exception


def test_telemetry_includes_timing_and_memory(runner):
    def workflow():
        data = [i * i for i in range(10)]
        return sum(data)

    metrics = runner.run(workflow)
    mod = metrics.modules[0]

    assert mod.result == 285
    assert mod.duration >= 0
    assert isinstance(mod.memory_before, int)
    assert isinstance(mod.memory_after, int)
    assert mod.memory_after >= mod.memory_before
    assert mod.memory_delta == mod.memory_after - mod.memory_before
    assert isinstance(mod.memory_peak, int)
    assert mod.memory_peak >= mod.memory_after

    telemetry = runner.telemetry
    assert telemetry is not None
    assert telemetry["memory_per_module"][mod.name] == mod.memory_delta


def test_module_specific_fixtures_restore_env(runner):
    values: list[str | None] = []

    def mod_one():
        values.append(os.getenv("TEST_ENV"))
        assert Path("data.txt").read_text() == "hello"

    def mod_two():
        values.append(os.getenv("TEST_ENV"))

    fixtures = {
        "mod_one": {"files": {"data.txt": "hello"}, "env": {"TEST_ENV": "one"}},
        "mod_two": {"env": {"TEST_ENV": "two"}},
    }

    runner.run([mod_one, mod_two], module_fixtures=fixtures)

    assert values == ["one", "two"]
    assert "TEST_ENV" not in os.environ
    assert runner.telemetry
    mods = runner.telemetry.get("module_fixtures", {})
    assert mods["mod_one"]["env"] == {"TEST_ENV": "one"}
    assert mods["mod_one"]["files"] == ["data.txt"]


def test_additional_fs_ops_confined_to_temp_dir(tmp_path, runner):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir3 = tmp_path / "dir3"
    tree = tmp_path / "tree"
    file1 = tmp_path / "file1"

    def workflow():
        os.makedirs(dir1.parent, exist_ok=True)
        os.mkdir(dir1)
        os.makedirs(dir2 / "sub")
        fd = os.open(file1, os.O_CREAT | os.O_WRONLY)
        os.write(fd, b"x")
        os.close(fd)
        os.stat(file1)
        os.rmdir(dir1)
        os.removedirs(dir2 / "sub")
        p = Path(dir3)
        p.mkdir()
        inner = p / "inner.txt"
        inner.write_text("hi")
        inner.unlink()
        p.rmdir()
        os.makedirs(tree / "sub")
        shutil.rmtree(tree)

    runner.run(workflow)

    for p in [dir1, dir2, dir3, tree, file1]:
        assert not p.exists()


def test_fs_mocks_for_new_wrappers(tmp_path, runner):
    calls: dict[str, str] = {}

    def rec(name: str):
        def _rec(path, *a, **kw):
            calls[name] = str(path)
            if name == "os.open":
                return 0
        return _rec

    fs_mocks = {
        name: rec(name)
        for name in [
            "os.mkdir",
            "os.makedirs",
            "os.rmdir",
            "os.removedirs",
            "os.open",
            "os.stat",
            "shutil.rmtree",
            "pathlib.Path.mkdir",
            "pathlib.Path.unlink",
            "pathlib.Path.rmdir",
        ]
    }

    paths = {
        "os.mkdir": tmp_path / "d1",
        "os.makedirs": tmp_path / "d2",
        "os.rmdir": tmp_path / "d3",
        "os.removedirs": tmp_path / "d4/sub",
        "os.open": tmp_path / "f5",
        "os.stat": tmp_path / "f6",
        "shutil.rmtree": tmp_path / "d7",
        "pathlib.Path.mkdir": tmp_path / "d8",
        "pathlib.Path.unlink": tmp_path / "f9",
        "pathlib.Path.rmdir": tmp_path / "d10",
    }

    def workflow():
        os.mkdir(paths["os.mkdir"])
        os.makedirs(paths["os.makedirs"])
        os.rmdir(paths["os.rmdir"])
        os.removedirs(paths["os.removedirs"])
        os.open(paths["os.open"], os.O_CREAT)
        os.stat(paths["os.stat"])
        shutil.rmtree(paths["shutil.rmtree"])
        Path(paths["pathlib.Path.mkdir"]).mkdir()
        Path(paths["pathlib.Path.unlink"]).unlink()
        Path(paths["pathlib.Path.rmdir"]).rmdir()

    runner.run(workflow, fs_mocks=fs_mocks)

    assert set(calls) == set(fs_mocks)
    for name, orig in paths.items():
        assert calls[name] != str(orig)


def test_safe_mode_blocks_directory_creation(runner):
    def workflow():
        os.mkdir("danger")

    metrics = runner.run(workflow, safe_mode=True)
    mod = metrics.modules[0]
    assert mod.success is False
    assert mod.exception and "file write disabled" in mod.exception


def test_tempfile_creation_confined_and_cleaned(runner):
    captured: dict[str, str | None] = {}

    def workflow():
        import tempfile

        tmpdir = Path(tempfile.gettempdir())

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x")
            captured["named"] = f.name
            assert Path(f.name).parent == tmpdir

        with tempfile.TemporaryFile() as f:
            name = getattr(f, "name", None)
            captured["temp"] = name if isinstance(name, str) else None
            if isinstance(name, str):
                assert Path(name).parent == tmpdir

        with tempfile.SpooledTemporaryFile(max_size=1) as f:
            f.write(b"xx")
            name = getattr(f, "name", None)
            captured["spooled"] = name if isinstance(name, str) else None
            if isinstance(name, str):
                assert Path(name).parent == tmpdir

    runner.run(workflow)

    for path in captured.values():
        if path:
            assert not Path(path).exists()
