import logging
import sandbox_runner.orphan_discovery as od


def test_cache_update_logs_error(tmp_path, monkeypatch, caplog):
    (tmp_path / "a.py").write_text("x = 1\n")  # path-ignore

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(od, "append_orphan_cache", boom)
    monkeypatch.setattr(od, "append_orphan_classifications", lambda *a, **k: None)
    caplog.set_level(logging.ERROR, logger=od.__name__)
    od.discover_recursive_orphans(str(tmp_path))
    assert "failed to update orphan cache" in caplog.text
