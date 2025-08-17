from tests.test_menace_master import _setup_mm_stubs
import sandbox_runner
import pytest


def test_meta_logger_basic(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    from tests.test_menace_master import _stub_module, DummyBot
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)
    log = sandbox_runner._SandboxMetaLogger(tmp_path / 'log.txt')
    log.log_cycle(0, 1.0, ['a.py'], 'first')
    log.log_cycle(1, 2.0, ['a.py', 'b.py'], 'second')
    log.log_cycle(2, 2.1, ['b.py'], 'third')

    ranking = dict(log.rankings())
    assert ranking['a.py'] == 1.5
    assert ranking['b.py'] == pytest.approx(0.6)
    assert log.diminishing() == []


def test_meta_logger_consecutive_threshold(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    from tests.test_menace_master import _stub_module, DummyBot
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)

    log = sandbox_runner._SandboxMetaLogger(tmp_path / "log.txt")
    threshold = 0.1
    log.log_cycle(0, 0.0, ["x.py"], "first")
    assert log.diminishing(threshold, consecutive=2) == []
    log.log_cycle(1, 0.05, ["x.py"], "second")
    assert log.diminishing(threshold, consecutive=2) == []
    log.log_cycle(2, 0.1, ["x.py"], "third")
    assert log.diminishing(threshold, consecutive=2) == ["x.py"]
    assert "x.py" in log.flagged_sections


def test_meta_logger_entropy_ceiling(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    from tests.test_menace_master import _stub_module, DummyBot
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)

    log = sandbox_runner._SandboxMetaLogger(tmp_path / "log.txt")
    log.log_cycle(0, 0.0, ["x.py"], "first", entropy_delta=0.1)
    log.log_cycle(1, 0.04, ["x.py"], "second", entropy_delta=0.2)
    log.log_cycle(2, 0.08, ["x.py"], "third", entropy_delta=0.2)

    assert log.ceiling(0.3, consecutive=2) == ["x.py"]
    assert "x.py" in log.flagged_sections

    new_log = sandbox_runner._SandboxMetaLogger(tmp_path / "log.txt")
    assert "x.py" in new_log.flagged_sections


def test_meta_logger_entropy_ceiling_no_flag_high_ratio(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    from tests.test_menace_master import _stub_module, DummyBot
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)

    log = sandbox_runner._SandboxMetaLogger(tmp_path / "log.txt")
    log.log_cycle(0, 0.0, ["x.py"], "first", entropy_delta=0.2)
    log.log_cycle(1, 0.6, ["x.py"], "second", entropy_delta=0.2)
    log.log_cycle(2, 1.2, ["x.py"], "third", entropy_delta=0.1)

    assert log.ceiling(1.0, consecutive=2) == []
    assert "x.py" not in log.flagged_sections


def test_meta_logger_entropy_ceiling_entropy_spike(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    from tests.test_menace_master import _stub_module, DummyBot
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)

    log = sandbox_runner._SandboxMetaLogger(tmp_path / "log.txt")
    log.log_cycle(0, 0.0, ["x.py"], "first", entropy_delta=0.05)
    log.log_cycle(1, 0.03, ["x.py"], "second", entropy_delta=0.2)
    log.log_cycle(2, 0.04, ["x.py"], "third", entropy_delta=0.3)

    assert log.ceiling(0.2, consecutive=2) == ["x.py"]
    assert "x.py" in log.flagged_sections
