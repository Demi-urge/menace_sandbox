from pathlib import Path

from patch_attempt_tracker import PatchAttemptTracker
from target_region import TargetRegion


class RecordingEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, TargetRegion | None]] = []

    def apply_patch(
        self, path: Path, desc: str, *, target_region: TargetRegion | None = None
    ):  # pragma: no cover - stub
        self.calls.append((path, target_region))
        text = path.read_text().splitlines()
        if target_region is None:
            path.write_text("# module patched\n")
        else:
            for i in range(target_region.start_line - 1, target_region.end_line):
                text[i] = "# patched"
            path.write_text("\n".join(text) + "\n")
        return None, False, 0.0


def test_escalation_and_reset():
    tracker = PatchAttemptTracker()
    filename = "mod" + ".py"  # path-ignore
    region = TargetRegion(start_line=1, end_line=2, function="f", filename=filename)

    level, _ = tracker.level_for(region, region)
    assert level == "region"
    tracker.record_failure(level, region, region)

    level, _ = tracker.level_for(region, region)
    tracker.record_failure(level, region, region)

    level, patch_region = tracker.level_for(region, region)
    assert level == "function" and patch_region == region
    tracker.record_failure(level, region, region)

    level, _ = tracker.level_for(region, region)
    tracker.record_failure(level, region, region)

    level, patch_region = tracker.level_for(region, region)
    assert level == "module" and patch_region is None

    tracker.reset(region)
    level, _ = tracker.level_for(region, region)
    assert level == "region"


def test_engine_respects_target_region(tmp_path):
    path = tmp_path / ("mod" + ".py")  # path-ignore
    path.write_text(
        "def f():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    return a + b\n"
    )

    engine = RecordingEngine()
    tracker = PatchAttemptTracker()
    region = TargetRegion(start_line=2, end_line=3, function="f", filename=str(path))

    for _ in range(2):
        level, patch_region = tracker.level_for(region, region)
        engine.apply_patch(path, "desc", target_region=patch_region)
        tracker.record_failure(level, region, region)

    level, patch_region = tracker.level_for(region, region)
    assert level == "function"
    engine.apply_patch(path, "desc", target_region=patch_region)
    lines = path.read_text().splitlines()
    assert lines[1] == "# patched" and lines[2] == "# patched"
    tracker.record_failure(level, region, region)

    level, patch_region = tracker.level_for(region, region)
    assert level == "function"
    engine.apply_patch(path, "desc", target_region=patch_region)
    tracker.record_failure(level, region, region)

    level, patch_region = tracker.level_for(region, region)
    assert level == "module" and patch_region is None
    engine.apply_patch(path, "desc", target_region=patch_region)
    assert path.read_text() == "# module patched\n"
