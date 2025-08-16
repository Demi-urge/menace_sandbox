import importlib
import json
import atexit
import types

import pytest


@pytest.fixture()
def radar_env(tmp_path, monkeypatch):
    """Provide an isolated relevancy radar instance with dummy modules."""

    # Prevent actual atexit handlers during tests
    monkeypatch.setattr(atexit, "register", lambda func: func)

    rr = importlib.import_module("relevancy_radar")

    usage_file = tmp_path / "module_usage.json"
    flags_file = tmp_path / "relevancy_flags.json"
    monkeypatch.setattr(rr, "_MODULE_USAGE_FILE", usage_file)
    monkeypatch.setattr(rr, "_RELEVANCY_FLAGS_FILE", flags_file)

    rr._module_usage_counter.clear()
    rr._relevancy_flags.clear()

    def make_mod(name: str):
        def run() -> None:
            rr.track_module_usage(name)

        return types.SimpleNamespace(run=run)

    modules = {name: make_mod(name) for name in ["alpha", "beta", "gamma"]}
    return rr, modules, usage_file, flags_file


def test_tracking_and_persistence(radar_env):
    """Ensure tracking increments counters and persistence merges counts."""

    rr, modules, usage_file, _ = radar_env

    modules["alpha"].run()
    modules["alpha"].run()
    modules["beta"].run()

    stats = rr.load_usage_stats()
    assert stats == {"alpha": 2, "beta": 1}

    rr._save_usage_counts()
    assert json.loads(usage_file.read_text()) == stats

    rr._module_usage_counter.clear()
    modules["alpha"].run()
    stats2 = rr.load_usage_stats()
    assert stats2 == {"alpha": 3, "beta": 1}


def test_relevancy_evaluation_unused_flagged(radar_env):
    """``evaluate_relevancy`` categorises modules and flags unused ones."""

    rr, modules, _, flags_file = radar_env

    for _ in range(6):
        modules["alpha"].run()
    modules["beta"].run()

    stats = rr.load_usage_stats()
    module_map = {name: 1 for name in modules}
    expected = {"alpha": "replace", "beta": "compress", "gamma": "retire"}

    assert rr.evaluate_relevancy(module_map, stats) == expected
    assert json.loads(flags_file.read_text()) == expected

    rr._relevancy_flags.clear()
    assert rr.flagged_modules() == expected
