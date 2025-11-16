import logging
import types
from pathlib import Path

from tests.test_recursive_orphans import _load_methods, DummyIndex


class DummyMetric:
    def inc(self, *_):
        pass


def test_workflow_ids_logged_and_stored(monkeypatch, tmp_path, caplog):
    _integrate_orphans, *_ = _load_methods()

    env = types.SimpleNamespace(
        auto_include_modules=lambda mods, recursive=True, validate=True: None,
        try_integrate_into_workflows=lambda mods: [11, 22],
    )
    g = _integrate_orphans.__globals__
    g['environment'] = env
    g['orphan_modules_legacy_total'] = DummyMetric()
    g['orphan_modules_redundant_total'] = DummyMetric()
    g['orphan_modules_reintroduced_total'] = DummyMetric()
    g['classify_module'] = lambda path: 'candidate'

    index = DummyIndex()
    logger = logging.getLogger('test_workflows')
    engine = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=logger,
        orphan_traces={},
        _last_orphan_counts={},
        _update_orphan_modules=lambda: None,
    )

    monkeypatch.setenv('SANDBOX_REPO_PATH', str(tmp_path))
    (tmp_path / 'mod.py').write_text('x = 1\n')  # path-ignore

    with caplog.at_level(logging.INFO):
        mods = _integrate_orphans(engine, [str(tmp_path / 'mod.py')])  # path-ignore

    assert mods == {'mod.py'}  # path-ignore
    assert engine.orphan_traces['mod.py']['workflows'] == [11, 22]  # path-ignore
    assert engine._last_orphan_counts['workflows_updated'] == 2
    assert any(getattr(record, 'workflows', None) == [11, 22] for record in caplog.records)
