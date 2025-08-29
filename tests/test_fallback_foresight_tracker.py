import os
import re
import textwrap
from pathlib import Path

from menace_sandbox.foresight_tracker import ForesightTracker as FullTracker


def _prepare_light_import_env(monkeypatch, tmp_path):
    """Create dummy system binaries and import the light ForesightTracker."""

    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ["ffmpeg", "tesseract", "qemu-system-x86_64"]:
        path = bindir / name
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    path = Path(__file__).resolve().parents[1] / "sandbox_runner.py"
    source = path.read_text()
    match = re.search(
        r"if os\.getenv\(\"MENACE_LIGHT_IMPORTS\"\):\n(?P<body>.+?)else:",
        source,
        re.DOTALL,
    )
    assert match is not None
    code = textwrap.dedent(match.group("body"))
    namespace: dict[str, object] = {}
    exec(code, namespace)
    return namespace["ForesightTracker"]


def test_full_tracker_window_property():
    tracker = FullTracker(max_cycles=5)
    assert tracker.window == 5


def test_light_import_tracker_roundtrip(monkeypatch, tmp_path):
    Fallback = _prepare_light_import_env(monkeypatch, tmp_path)

    tracker = Fallback(window=2)
    tracker.record_cycle_metrics("wf", {"m": 1})
    tracker.record_cycle_metrics("wf", {"m": 2})

    data = tracker.to_dict()
    assert data["window"] == 2
    assert [e["m"] for e in data["history"]["wf"]] == [1.0, 2.0]

    loaded_full = FullTracker.from_dict(data, max_cycles=2)
    assert [e["m"] for e in loaded_full.history["wf"]] == [1.0, 2.0]

    loaded_fallback = Fallback.from_dict(loaded_full.to_dict(), window=2)
    assert [e["m"] for e in loaded_fallback.history["wf"]] == [1.0, 2.0]

