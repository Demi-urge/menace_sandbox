import sys
import types

stub_cycle = types.ModuleType("sandbox_runner.cycle")
stub_cycle._async_track_usage = lambda *a, **k: None
sys.modules["sandbox_runner.cycle"] = stub_cycle

from sandbox_runner.meta_logger import _SandboxMetaLogger


def test_entropy_ceiling_flags_and_halts(tmp_path):
    log = _SandboxMetaLogger(tmp_path / "meta.log")

    roi_values = [0.0, 0.1, 0.15, 0.2]
    cycles = 0

    for roi in roi_values:
        log.log_cycle(cycles, roi, ["m.py"], "cycle")  # path-ignore
        cycles += 1
        if log.ceiling(0.3, consecutive=2):
            break

    assert "m.py" in log.flagged_sections  # path-ignore
    assert cycles == 4

