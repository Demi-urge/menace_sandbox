from __future__ import annotations


def test_legacy_compat_reexports_importable():
    from coding_bot_interface import prepare_pipeline_for_bootstrap
    from menace.data_bot import MetricsDB
    from menace.diagnostic_manager import ContextBuilder
    from menace.self_coding_manager import DataBot, SelfCodingManager

    assert prepare_pipeline_for_bootstrap is not None
    assert ContextBuilder is not None
    assert DataBot is not None
    assert SelfCodingManager is not None
    assert MetricsDB is not None
