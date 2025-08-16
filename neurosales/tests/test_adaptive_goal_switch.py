import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.adaptive_goal_switch import AdaptiveGoalSwitcher


def test_switch_from_convert_to_comfort():
    watcher = AdaptiveGoalSwitcher(start_goal="convert", window=2, stickiness=1)
    watcher.update_metrics(0.2, 0.8, 0.3)
    watcher.update_metrics(0.1, 0.9, 0.2)
    assert watcher.current_goal == "comfort"
    assert len(watcher.logs) == 1


def test_stickiness_prevents_flip_flop():
    watcher = AdaptiveGoalSwitcher(start_goal="convert", window=1, stickiness=3)
    for _ in range(2):
        watcher.update_metrics(0.1, 0.9, 0.1)
    # stickiness not reached yet
    assert watcher.current_goal == "convert"
    watcher.update_metrics(0.1, 0.9, 0.1)
    assert watcher.current_goal == "comfort"


def test_explain_last():
    watcher = AdaptiveGoalSwitcher(window=2, stickiness=1)
    for _ in range(2):
        watcher.update_metrics(0.2, 0.8, 0.2)
    msg = watcher.explain_last()
    assert "convert" in msg and "comfort" in msg
