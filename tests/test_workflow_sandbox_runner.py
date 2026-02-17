import importlib.util
import json
import os
import sys
import tempfile
import types
import builtins
import logging
from pathlib import Path

import urllib.request
import shutil
import multiprocessing

import pytest
import time

from dynamic_path_router import resolve_dir, resolve_path, repo_root

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Dynamically load WorkflowSandboxRunner without importing the heavy package
package_path = resolve_dir("sandbox_runner")
package = types.ModuleType("sandbox_runner")
package.__path__ = [str(package_path)]
sys.modules["sandbox_runner"] = package

spec = importlib.util.spec_from_file_location(
    "sandbox_runner.workflow_sandbox_runner", resolve_path("workflow_sandbox_runner.py")  # path-ignore
)
wsr = importlib.util.module_from_spec(spec)
assert spec.loader
sys.modules[spec.name] = wsr
spec.loader.exec_module(wsr)
WorkflowSandboxRunner = wsr.WorkflowSandboxRunner
EmptyWorkflowError = wsr.EmptyWorkflowError

from tests.fixtures.workflow_modules import mod_a  # noqa: E402


@pytest.fixture(autouse=True)
def _no_psutil(monkeypatch):
    """Ensure psutil is absent so tests don't depend on it."""
    monkeypatch.setattr(wsr, "psutil", None, raising=False)


def test_empty_workflow_errors():
    runner = WorkflowSandboxRunner()
    with pytest.raises(EmptyWorkflowError) as exc:
        runner.run([])
    assert "contained no actionable steps" in str(exc.value)


def test_files_confined_to_temp_dir(monkeypatch):
    outside = Path("/tmp/outside_file.txt")
    if outside.exists():
        outside.unlink()

    captured: list[Path] = []
    original_write = Path.write_text

    def record_write(path, data, *a, **kw):
        p = Path(path)
        captured.append(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        return original_write(p, data, *a, **kw)

    def writer_step():
        outside.write_text("data")

    runner = WorkflowSandboxRunner()
    runner.run(
        [mod_a.start, writer_step],
        safe_mode=True,
        fs_mocks={"pathlib.Path.write_text": record_write},
    )

    assert not outside.exists()
    assert captured, "file write should have been captured"
    assert captured[0] != outside
    assert str(captured[0]).startswith(tempfile.gettempdir())


def test_safe_mode_blocks_network_and_reports_telemetry():
    def network_step():
        urllib.request.urlopen("http://example.com")

    runner = WorkflowSandboxRunner()
    metrics = runner.run([mod_a.start, network_step], safe_mode=True)

    assert metrics.crash_count == 1
    assert len(metrics.modules) == 2
    assert metrics.modules[1].success is False
    assert "network access disabled" in (metrics.modules[1].exception or "")

    telemetry = runner.telemetry
    assert telemetry is not None
    assert set(telemetry["time_per_module"]) == {"start", "network_step"}
    assert set(telemetry["cpu_per_module"]) == {"start", "network_step"}
    for ct in telemetry["cpu_per_module"].values():
        assert ct >= 0
    assert telemetry["crash_frequency"] == pytest.approx(1 / 2)


def test_httpx_and_fs_wrappers():
    src = Path("/tmp/httpx_src.txt")
    dst = Path("/tmp/httpx_dst.txt")
    src.write_text("outside")
    if dst.exists():
        dst.unlink()

    def step():
        import httpx

        shutil.copy(str(src), str(dst))
        os.remove(str(src))
        with httpx.Client() as client:
            client.get("http://example.com")

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True, test_data={str(src): "inside"})

    assert metrics.crash_count == 1
    assert src.exists()
    assert not dst.exists()

    src.unlink()
    if dst.exists():
        dst.unlink()


def test_httpx_network_mock():
    httpx = pytest.importorskip("httpx")

    def step():
        import httpx
        resp = httpx.get("http://example.com")
        return resp.text

    runner = WorkflowSandboxRunner()
    metrics = runner.run(
        [step],
        safe_mode=True,
        network_mocks={
            "http://example.com": (
                lambda self, method, url, *a, **kw: httpx.Response(200, text="mocked")
            )
        },
    )

    assert metrics.crash_count == 0
    assert metrics.modules[0].result == "mocked"




def test_subprocess_guard_forbids_absolute_open_outside_root():
    def step():
        with open('/root/forbidden', 'r'):
            pass

    runner = WorkflowSandboxRunner()
    metrics = runner.run(
        [step],
        safe_mode=True,
        use_subprocess=False,
        subprocess_guard=True,
    )

    assert metrics.crash_count == 1
    exc = metrics.modules[0].exception or ''
    assert 'forbidden path access' in exc
    assert 'error: forbidden path access' in exc
    assert 'No such file or directory' not in exc


@pytest.mark.parametrize(
    'step',
    [
        lambda: Path('/root/forbidden').open('r'),
        lambda: Path('/root/forbidden').write_text('x'),
        lambda: Path('/root/forbidden').write_bytes(b'x'),
    ],
)
def test_subprocess_guard_path_wrappers_report_forbidden_path_access(step):
    runner = WorkflowSandboxRunner()
    metrics = runner.run(
        [step],
        safe_mode=True,
        use_subprocess=False,
        subprocess_guard=True,
    )

    assert metrics.crash_count == 1
    exc = metrics.modules[0].exception or ''
    assert 'forbidden path access' in exc
    assert 'error: forbidden path access' in exc
    assert 'No such file or directory' not in exc

def test_open_write_requires_mock_in_safe_mode():
    def step():
        with open("foo.txt", "w") as f:
            f.write("data")

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True)
    assert metrics.crash_count == 1
    assert "file write disabled in safe_mode" in (metrics.modules[0].exception or "")


def test_path_write_text_requires_mock_in_safe_mode():
    def step():
        Path("foo.txt").write_text("data")

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True)
    assert metrics.crash_count == 1
    assert "file write disabled in safe_mode" in (metrics.modules[0].exception or "")


def test_path_write_bytes_requires_mock_in_safe_mode():
    def step():
        Path("foo.bin").write_bytes(b"data")

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True)
    assert metrics.crash_count == 1
    assert "file write disabled in safe_mode" in (metrics.modules[0].exception or "")


def test_mocked_writes_succeed_in_safe_mode():
    original_open = builtins.open
    captured: list[Path] = []

    def mock_open(path, mode, *a, **kw):
        captured.append(Path(path))
        return original_open(path, mode, *a, **kw)

    def step():
        with open("foo.txt", "w") as f:
            f.write("ok")

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True, fs_mocks={"open": mock_open})

    assert metrics.crash_count == 0
    assert captured


def test_aiohttp_blocked_in_safe_mode():
    pytest.importorskip("aiohttp")

    def step():
        import asyncio
        import aiohttp  # noqa: F401

        async def inner():
            async with aiohttp.ClientSession() as session:
                await session.get("http://example.com")

        asyncio.run(inner())

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True)

    assert metrics.crash_count == 1
    assert "network access disabled" in (metrics.modules[0].exception or "")


def test_aiohttp_network_mock():
    pytest.importorskip("aiohttp")

    async def mock_request(self, method, url, *a, **kw):  # pragma: no cover - invoked by patch
        class Resp:
            status = 200

            async def read(self) -> bytes:
                return b"mocked"

            async def text(self) -> str:
                return "mocked"

        return Resp()

    def step():
        import asyncio
        import aiohttp  # noqa: F401

        async def inner():
            async with aiohttp.ClientSession() as session:
                resp = await session.get("http://example.com")
                return await resp.text()

        return asyncio.run(inner())

    runner = WorkflowSandboxRunner()
    metrics = runner.run(
        [step],
        safe_mode=True,
        network_mocks={"http://example.com": mock_request},
    )

    assert metrics.crash_count == 0
    assert metrics.modules[0].result == "mocked"


def test_socket_blocked_in_safe_mode():
    def step():
        import socket  # noqa: F401

        socket.socket()

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True)

    assert metrics.crash_count == 1
    assert "network access disabled" in (metrics.modules[0].exception or "")


def test_os_shutil_wrappers_redirected():
    src_file = Path("/tmp/wrapper_src.txt")
    src_dir = Path("/tmp/wrapper_dir")
    src_dir_file = src_dir / "data.txt"

    src_file.write_text("outside")
    src_dir_file.parent.mkdir(parents=True, exist_ok=True)
    src_dir_file.write_text("outside_dir")

    copy2_dst = Path("/tmp/wrapper_copy2.txt")
    copyfile_dst = Path("/tmp/wrapper_copyfile.txt")
    move_dst = Path("/tmp/wrapper_move.txt")
    rename_dst = Path("/tmp/wrapper_rename.txt")
    replace_dst = Path("/tmp/wrapper_replace.txt")
    copytree_dst = Path("/tmp/wrapper_copytree")

    for p in [
        copy2_dst,
        copyfile_dst,
        move_dst,
        rename_dst,
        replace_dst,
        copytree_dst,
    ]:
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    def step():
        shutil.copy2(src_file, copy2_dst)
        shutil.copyfile(src_file, copyfile_dst)
        shutil.copytree(src_dir, copytree_dst)
        shutil.move(copyfile_dst, move_dst)
        os.rename(move_dst, rename_dst)
        os.replace(rename_dst, replace_dst)
        os.unlink(src_file)

    runner = WorkflowSandboxRunner()
    runner.run(
        [step],
        safe_mode=True,
        test_data={str(src_file): "inside", str(src_dir_file): "inside_dir"},
    )

    assert src_file.exists()
    assert src_dir.exists() and src_dir_file.exists()
    for p in [
        copy2_dst,
        copyfile_dst,
        move_dst,
        rename_dst,
        replace_dst,
        copytree_dst,
    ]:
        assert not p.exists()

    src_file.unlink()
    src_dir_file.unlink()
    src_dir.rmdir()


def test_callable_results_captured():
    def step1():
        return "alpha"

    def step2():
        return 123

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step1, step2], safe_mode=True)

    assert [m.result for m in metrics.modules] == ["alpha", 123]

    telemetry = runner.telemetry
    assert telemetry is not None
    assert telemetry["results"]["step1"] == "alpha"
    assert telemetry["results"]["step2"] == 123


def test_json_responses_from_requests_and_urllib():
    pytest.importorskip("requests")

    def step_requests():
        import requests

        return requests.get("http://example.com/data").json()

    def step_urllib():
        import urllib.request

        return urllib.request.urlopen("http://example.com/data").json()

    runner = WorkflowSandboxRunner()
    payload = {"msg": "ok"}
    metrics = runner.run(
        [step_requests, step_urllib],
        safe_mode=True,
        test_data={"http://example.com/data": json.dumps(payload)},
    )

    assert metrics.modules[0].result == payload
    assert metrics.modules[1].result == payload


def test_async_workflow_metrics():
    import asyncio

    async def ok():
        await asyncio.sleep(0)
        return "ok"

    def returns_coroutine():
        async def inner():
            await asyncio.sleep(0)
            return "inner"

        return inner()

    async def crash():
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    runner = WorkflowSandboxRunner()
    metrics = runner.run([ok, returns_coroutine, crash], safe_mode=True)

    assert len(metrics.modules) == 3
    assert metrics.modules[0].result == "ok"
    assert metrics.modules[1].result == "inner"
    assert metrics.modules[2].success is False
    assert "boom" in (metrics.modules[2].exception or "")
    assert metrics.crash_count == 1

    for mod in metrics.modules:
        assert isinstance(mod.duration, float)
        assert isinstance(mod.cpu_before, float)
        assert isinstance(mod.cpu_after, float)
        assert isinstance(mod.cpu_delta, float)
        assert mod.cpu_delta >= 0
        assert isinstance(mod.memory_before, int)
        assert isinstance(mod.memory_after, int)
        assert isinstance(mod.memory_delta, int)
        assert isinstance(mod.memory_peak, int)
        assert mod.memory_peak >= mod.memory_after

    telemetry = runner.telemetry
    assert telemetry is not None
    assert set(telemetry["time_per_module"]) == {"ok", "returns_coroutine", "crash"}
    assert set(telemetry["memory_per_module"]) == {"ok", "returns_coroutine", "crash"}
    assert set(telemetry["peak_memory_per_module"]) == {
        "ok",
        "returns_coroutine",
        "crash",
    }
    for mod in metrics.modules:
        assert telemetry["memory_per_module"][mod.name] == mod.memory_delta
        assert telemetry["peak_memory_per_module"][mod.name] == mod.memory_peak
    assert telemetry["crash_frequency"] == pytest.approx(1 / 3)


def test_module_specific_fixtures_apply_files_and_env():
    """Per-module fixtures should inject files and environment variables."""

    os.environ.pop("WF_VAR", None)
    recorded: list[str | None] = []

    def one() -> None:
        recorded.append(Path("fixture.txt").read_text())
        recorded.append(os.getenv("WF_VAR"))

    def two() -> None:
        recorded.append(os.getenv("WF_VAR"))

    fixtures = {
        "one": {"files": {"fixture.txt": "hello"}, "env": {"WF_VAR": "A"}},
        "two": {"env": {"WF_VAR": "B"}},
    }

    runner = WorkflowSandboxRunner()
    runner.run([one, two], module_fixtures=fixtures)

    assert recorded == ["hello", "A", "B"]
    assert "WF_VAR" not in os.environ

    telemetry = runner.telemetry or {}
    mods = telemetry.get("module_fixtures", {})
    assert mods["one"]["files"] == ["fixture.txt"]
    assert mods["one"]["env"] == {"WF_VAR": "A"}


def test_timeout_aborts_module():
    runner = WorkflowSandboxRunner()

    def slow():
        time.sleep(5)

    metrics = runner.run([slow], safe_mode=True, timeout=0.2)

    assert metrics.crash_count == 1
    mod = metrics.modules[0]
    assert mod.success is False
    assert "timeout" in (mod.exception or "").lower()


def test_memory_limit_aborts_module(monkeypatch):
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(wsr, "psutil", psutil, raising=False)

    runner = WorkflowSandboxRunner()

    def eater():
        data = []
        while True:
            data.append(bytearray(1024 * 1024))

    metrics = runner.run([eater], safe_mode=True, memory_limit=5 * 1024 * 1024)

    assert metrics.crash_count == 1
    mod = metrics.modules[0]
    assert mod.success is False
    assert "memory" in (mod.exception or "").lower()


def test_module_copy_failure_logged(monkeypatch, caplog):
    def step():
        return "ok"

    def boom(*a, **kw):
        raise OSError("copy boom")

    monkeypatch.setattr(shutil, "copy2", boom)
    runner = WorkflowSandboxRunner()

    with caplog.at_level(logging.WARNING):
        runner.run([step])

    assert any("failed to copy source" in r.message for r in caplog.records)


def test_telemetry_write_error_logged(monkeypatch, caplog):
    def step():
        return "ok"

    original = Path.write_text

    def bad_write(self, *a, **kw):
        if self.name == "telemetry.json":
            raise OSError("telemetry boom")
        return original(self, *a, **kw)

    monkeypatch.setattr(Path, "write_text", bad_write)
    runner = WorkflowSandboxRunner()

    with caplog.at_level(logging.ERROR):
        runner.run([step])

    assert any("failed to persist telemetry" in r.message for r in caplog.records)


def test_module_failure_logs_and_cleans(monkeypatch, caplog):
    paths: dict[str, str] = {}
    original_tmpdir = tempfile.TemporaryDirectory

    def record_tmpdir(*a, **kw):
        td = original_tmpdir(*a, **kw)
        paths["path"] = td.name
        return td

    monkeypatch.setattr(tempfile, "TemporaryDirectory", record_tmpdir)
    orig_open = builtins.open

    def failing_step():
        raise RuntimeError("boom")

    runner = WorkflowSandboxRunner()
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            runner.run([failing_step])

    assert any("module failing_step failed" in r.message for r in caplog.records)
    assert builtins.open is orig_open
    assert paths["path"]
    assert not Path(paths["path"]).exists()


def test_subprocess_failure_cleanup(caplog):
    def failing_step():
        raise RuntimeError("boom")

    runner = WorkflowSandboxRunner()
    before = set(multiprocessing.active_children())
    with caplog.at_level(logging.ERROR):
        metrics = runner.run(failing_step, use_subprocess=True)
    after = set(multiprocessing.active_children())

    assert before == after
    assert runner.telemetry and "error" in runner.telemetry
    assert any("subprocess error" in r.message for r in caplog.records)
