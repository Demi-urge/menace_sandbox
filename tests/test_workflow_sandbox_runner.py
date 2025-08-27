import importlib.util
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
