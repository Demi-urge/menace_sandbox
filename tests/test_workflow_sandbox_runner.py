import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import urllib.request
import shutil

import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Dynamically load WorkflowSandboxRunner without importing the heavy package
package_path = Path(__file__).resolve().parent.parent / "sandbox_runner"
package = types.ModuleType("sandbox_runner")
package.__path__ = [str(package_path)]
sys.modules["sandbox_runner"] = package

spec = importlib.util.spec_from_file_location(
    "sandbox_runner.workflow_sandbox_runner",
    package_path / "workflow_sandbox_runner.py",
)
wsr = importlib.util.module_from_spec(spec)
assert spec.loader
sys.modules[spec.name] = wsr
spec.loader.exec_module(wsr)
WorkflowSandboxRunner = wsr.WorkflowSandboxRunner

from tests.fixtures.workflow_modules import mod_a  # noqa: E402


@pytest.fixture(autouse=True)
def _no_psutil(monkeypatch):
    """Ensure psutil is absent so tests don't depend on it."""
    monkeypatch.setattr(wsr, "psutil", None, raising=False)


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
        network_mocks={"httpx": lambda self, method, url, *a, **kw: httpx.Response(200, text="mocked")},
    )

    assert metrics.crash_count == 0
    assert metrics.modules[0].result == "mocked"


def test_aiohttp_blocked_in_safe_mode():
    aiohttp = pytest.importorskip("aiohttp")

    def step():
        import asyncio, aiohttp  # noqa: F401

        async def inner():
            async with aiohttp.ClientSession() as session:
                await session.get("http://example.com")

        asyncio.run(inner())

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True)

    assert metrics.crash_count == 1
    assert "network access disabled" in (metrics.modules[0].exception or "")


def test_aiohttp_network_mock():
    aiohttp = pytest.importorskip("aiohttp")

    async def mock_request(self, method, url, *a, **kw):  # pragma: no cover - invoked by patch
        class Resp:
            status = 200

            async def read(self) -> bytes:
                return b"mocked"

            async def text(self) -> str:
                return "mocked"

        return Resp()

    def step():
        import asyncio, aiohttp  # noqa: F401

        async def inner():
            async with aiohttp.ClientSession() as session:
                resp = await session.get("http://example.com")
                return await resp.text()

        return asyncio.run(inner())

    runner = WorkflowSandboxRunner()
    metrics = runner.run(
        [step],
        safe_mode=True,
        network_mocks={"aiohttp": mock_request},
    )

    assert metrics.crash_count == 0
    assert metrics.modules[0].result == "mocked"


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
