from tests.test_menace_master import _setup_mm_stubs
import sandbox_runner
import pytest
from sandbox_runner.meta_logger import _SandboxMetaLogger


def test_meta_logger_basic(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    from tests.test_menace_master import _stub_module, DummyBot
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)
    log = _SandboxMetaLogger(tmp_path / 'log.txt')
    log.log_cycle(0, 1.0, ['a.py'], 'first')  # path-ignore
    log.log_cycle(1, 2.0, ['a.py', 'b.py'], 'second')  # path-ignore
    log.log_cycle(2, 2.1, ['b.py'], 'third')  # path-ignore

    ranking = log.rankings()
    assert ranking[0][:2] == (2, 2.1)
    assert ranking[-1][:2] == (0, 1.0)
    assert log.diminishing() == []


def test_meta_logger_consecutive_threshold(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    from tests.test_menace_master import _stub_module, DummyBot
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)

    log = _SandboxMetaLogger(tmp_path / "log.txt")
    threshold = 0.1
    log.log_cycle(0, 0.0, ["x.py"], "first")  # path-ignore
    assert log.diminishing(threshold, consecutive=2, entropy_threshold=-1.0) == []
    log.log_cycle(1, 0.05, ["x.py"], "second")  # path-ignore
    assert log.diminishing(threshold, consecutive=2, entropy_threshold=-1.0) == []
    log.log_cycle(2, 0.1, ["x.py"], "third")  # path-ignore
    assert log.diminishing(threshold, consecutive=2, entropy_threshold=-1.0) == ["x.py"]  # path-ignore
    assert "x.py" in log.flagged_sections  # path-ignore


def test_meta_logger_entropy_ceiling(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    from tests.test_menace_master import _stub_module, DummyBot
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)

    log = _SandboxMetaLogger(tmp_path / "log.txt")
    log.log_cycle(0, 0.0, ["x.py"], "first")  # path-ignore
    log.log_cycle(1, 0.1, ["x.py"], "second")  # path-ignore
    log.log_cycle(2, 0.15, ["x.py"], "third")  # path-ignore
    log.log_cycle(3, 0.2, ["x.py"], "fourth")  # path-ignore

    assert log.ceiling(0.3, consecutive=2) == ["x.py"]  # path-ignore
    assert "x.py" in log.flagged_sections  # path-ignore

    new_log = _SandboxMetaLogger(tmp_path / "log.txt")
    assert "x.py" in new_log.flagged_sections  # path-ignore


def test_meta_logger_entropy_ceiling_no_flag_high_ratio(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    from tests.test_menace_master import _stub_module, DummyBot
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)

    log = _SandboxMetaLogger(tmp_path / "log.txt")
    log.log_cycle(0, 0.0, ["x.py"], "first")  # path-ignore
    log.log_cycle(1, 1.0, ["x.py"], "second")  # path-ignore
    log.log_cycle(2, 2.0, ["x.py"], "third")  # path-ignore
    log.log_cycle(3, 3.0, ["x.py"], "fourth")  # path-ignore

    assert log.ceiling(1.0, consecutive=2) == []
    assert "x.py" not in log.flagged_sections  # path-ignore


def test_meta_logger_entropy_ceiling_entropy_spike(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    from tests.test_menace_master import _stub_module, DummyBot
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)

    log = _SandboxMetaLogger(tmp_path / "log.txt")
    log.log_cycle(0, 0.0, ["x.py"], "first")  # path-ignore
    log.log_cycle(1, 0.03, ["x.py"], "second")  # path-ignore
    log.log_cycle(2, 0.06, ["x.py"], "third")  # path-ignore
    log.log_cycle(3, 0.09, ["x.py"], "fourth")  # path-ignore

    assert log.ceiling(0.2, consecutive=2) == ["x.py"]  # path-ignore
    assert "x.py" in log.flagged_sections  # path-ignore


def test_meta_logger_entropy_plateau(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    from tests.test_menace_master import _stub_module, DummyBot
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)

    log = _SandboxMetaLogger(tmp_path / "log.txt")
    rois = [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5]
    for idx, roi in enumerate(rois):
        log.log_cycle(idx, roi, ["x.py"], str(idx))  # path-ignore

    assert log.diminishing(0.1, consecutive=2, entropy_threshold=0.01) == ["x.py"]  # path-ignore
    assert "x.py" in log.flagged_sections  # path-ignore


def test_meta_logger_entropy_persistence(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    from tests.test_menace_master import _stub_module, DummyBot
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorDB=lambda p: DummyBot(), ErrorBot=DummyBot)

    path = tmp_path / "log.txt"
    log = _SandboxMetaLogger(path)
    log.log_cycle(0, 0.0, ["m.py"], "first")  # path-ignore
    log.log_cycle(1, 0.1, ["m.py"], "second")  # path-ignore
    ent_before = log.module_entropy_deltas["m.py"][-1]  # path-ignore

    new_log = _SandboxMetaLogger(path)
    assert new_log.module_entropy_deltas["m.py"][-1] == pytest.approx(ent_before)  # path-ignore
    new_log.log_cycle(2, 0.15, ["m.py"], "third")  # path-ignore
    assert len(new_log.module_entropy_deltas["m.py"]) == 3  # path-ignore
