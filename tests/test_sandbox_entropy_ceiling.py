import menace.roi_tracker as rt
from sandbox_runner.meta_logger import _SandboxMetaLogger


def test_entropy_ceiling_flags_and_halts(tmp_path):
    tracker = rt.ROITracker(entropy_threshold=0.5)
    log = _SandboxMetaLogger(tmp_path / "meta.log")

    # initialise baseline entropy
    tracker.update(0.0, 0.0, metrics={"synergy_shannon_entropy": 0.0})

    roi_values = [1.0, 1.2, 1.25, 1.3]
    entropy_values = [1.0, 2.0, 3.0, 4.0]
    prev_roi = 0.0
    cycles = 0

    for roi, ent in zip(roi_values, entropy_values):
        _, _, _, ceiling = tracker.update(
            prev_roi, roi, ["m.py"], metrics={"synergy_shannon_entropy": ent}
        )
        ent_delta = tracker.entropy_delta_history[-1]
        log.log_cycle(cycles, roi, ["m.py"], "cycle", entropy_delta=ent_delta)
        cycles += 1
        if ceiling:
            log.ceiling(tracker.entropy_threshold)
            break
        prev_roi = roi

    assert "m.py" in log.flagged_sections
    assert cycles == 3

