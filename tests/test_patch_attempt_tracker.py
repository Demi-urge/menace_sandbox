import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec_tr = importlib.util.spec_from_file_location(
    "menace.self_improvement.target_region",
    ROOT / "self_improvement" / "target_region.py",  # path-ignore
)
tr_module = importlib.util.module_from_spec(_spec_tr)
sys.modules.setdefault("menace.self_improvement.target_region", tr_module)
_spec_tr.loader.exec_module(tr_module)  # type: ignore[attr-defined]
TargetRegion = tr_module.TargetRegion

_spec_pat = importlib.util.spec_from_file_location(
    "menace.patch_attempt_tracker", ROOT / "patch_attempt_tracker.py"  # path-ignore
)
pat_module = importlib.util.module_from_spec(_spec_pat)
sys.modules.setdefault("menace.patch_attempt_tracker", pat_module)
_spec_pat.loader.exec_module(pat_module)  # type: ignore[attr-defined]
PatchAttemptTracker = pat_module.PatchAttemptTracker


class DummyLogger:
    def __init__(self):
        self.events = []

    def info(self, msg, extra=None):  # pragma: no cover - simple recorder
        self.events.append((msg, extra or {}))


def _apply(path: Path, region: TargetRegion | None) -> None:
    """Apply a patch by replacing the region with a marker."""
    lines = path.read_text().splitlines()
    if region is None:
        path.write_text("# module patched\n")
    else:
        for i in range(region.start_line - 1, region.end_line):
            lines[i] = "# patched"
        path.write_text("\n".join(lines) + "\n")


def test_escalation_from_region_to_module(tmp_path):
    path = tmp_path / "mod.py"  # path-ignore
    path.write_text(
        "def f():\n    a=1\n    b=2\n    return a+b\n\n"
        "def g():\n    return 42\n"
    )

    region = TargetRegion(filename=str(path), start_line=2, end_line=2, function="f")
    func_region = TargetRegion(filename=str(path), start_line=1, end_line=4, function="f")

    logger = DummyLogger()
    tracker = PatchAttemptTracker(logger=logger)

    # First attempt: region level patch
    level, target = tracker.level_for(region, func_region)
    assert level == "region" and target == region
    _apply(path, target)
    lines = path.read_text().splitlines()
    assert lines[1] == "# patched"
    assert lines[2] == "    b=2"
    assert lines[5] == "def g():"  # other function untouched
    tracker.record_failure(level, region, func_region)

    # Second attempt: still region level, escalation triggered afterwards
    level, target = tracker.level_for(region, func_region)
    assert level == "region"
    _apply(path, target)
    tracker.record_failure(level, region, func_region)
    assert logger.events[0][1]["level"] == "function"

    # Third attempt: function level patch
    level, target = tracker.level_for(region, func_region)
    assert level == "function" and target == func_region
    _apply(path, target)
    lines = path.read_text().splitlines()
    assert lines[0:4] == ["# patched"] * 4
    assert lines[5] == "def g():"  # other function still untouched
    tracker.record_failure(level, region, func_region)

    # Fourth attempt: function level again, escalation to module
    level, target = tracker.level_for(region, func_region)
    assert level == "function"
    _apply(path, target)
    tracker.record_failure(level, region, func_region)
    assert logger.events[1][1]["level"] == "module"

    # Fifth attempt: module level rewrite
    level, target = tracker.level_for(region, func_region)
    assert level == "module" and target is None
    _apply(path, target)
    assert path.read_text() == "# module patched\n"


def test_reset_clears_escalation(tmp_path):
    path = tmp_path / "mod.py"  # path-ignore
    path.write_text("def f():\n    pass\n")
    region = TargetRegion(filename=str(path), start_line=1, end_line=1, function="f")
    func_region = TargetRegion(filename=str(path), start_line=1, end_line=2, function="f")

    logger = DummyLogger()
    tracker = PatchAttemptTracker(logger=logger)

    tracker.record_failure("region", region, func_region)
    tracker.record_failure("region", region, func_region)
    assert tracker.level_for(region, func_region)[0] == "function"
    assert logger.events[0][1]["level"] == "function"

    # Simulate successful patch and reset counters
    tracker.reset(region)
    level, _ = tracker.level_for(region, func_region)
    assert level == "region"

    tracker.record_failure("region", region, func_region)
    level, _ = tracker.level_for(region, func_region)
    assert level == "region"  # escalation count reset
    assert len(logger.events) == 1  # no new escalation logged


def test_module_level_after_two_function_failures(tmp_path):
    path = tmp_path / "mod.py"  # path-ignore
    path.write_text("def f():\n    a=1\n    b=2\n    return a+b\n")

    region = TargetRegion(filename=str(path), start_line=2, end_line=2, function="f")
    func_region = TargetRegion(filename=str(path), start_line=1, end_line=4, function="f")

    tracker = PatchAttemptTracker(logger=DummyLogger())

    tracker.record_failure("region", region, func_region)
    tracker.record_failure("region", region, func_region)
    assert tracker.level_for(region, func_region)[0] == "function"

    tracker.record_failure("function", region, func_region)
    tracker.record_failure("function", region, func_region)
    level, target = tracker.level_for(region, func_region)
    assert level == "module" and target is None
