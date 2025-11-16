import os
import sys
import types

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

stub_cycle = types.ModuleType("sandbox_runner.cycle")
stub_cycle._async_track_usage = lambda *a, **k: None
sys.modules["sandbox_runner.cycle"] = stub_cycle

from sandbox_runner.meta_logger import _SandboxMetaLogger


def _log_rois(log, rois):
    for idx, roi in enumerate(rois):
        log.log_cycle(idx, float(roi), ["m.py"], "cycle")  # path-ignore


def test_diminishing_entropy_convergence_flags(tmp_path):
    log = _SandboxMetaLogger(tmp_path / "meta.log")
    _log_rois(log, range(10))
    assert log.diminishing(0.1, consecutive=3, entropy_threshold=0.01) == ["m.py"]  # path-ignore
    assert "m.py" in log.flagged_sections  # path-ignore


def test_diminishing_entropy_above_threshold_not_flagged(tmp_path):
    log = _SandboxMetaLogger(tmp_path / "meta.log")
    _log_rois(log, range(6))
    assert log.diminishing(0.1, consecutive=3, entropy_threshold=0.01) == []
    assert "m.py" not in log.flagged_sections  # path-ignore
