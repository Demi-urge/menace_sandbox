import builtins
import logging
import types

import menace.weekly_report_generator as wrg

def test_parse_timestamp_logs_warning(caplog):
    caplog.set_level(logging.WARNING)
    caplog.clear()
    wrg._parse_timestamp("bad")
    assert any('failed to parse' in rec.message for rec in caplog.records)

def test_send_to_discord_logs_exception(monkeypatch, caplog, tmp_path):
    caplog.set_level(logging.ERROR)

    monkeypatch.setattr(wrg, 'requests', types.SimpleNamespace(post=lambda *a, **k: None))
    report = tmp_path / 'report.txt'
    report.write_text('x')

    def fake_open(*a, **k):
        raise RuntimeError('boom')

    monkeypatch.setattr(builtins, 'open', fake_open)
    wrg.send_to_discord(str(report), 'http://example.com')
    assert any('failed to read report' in rec.message for rec in caplog.records)

