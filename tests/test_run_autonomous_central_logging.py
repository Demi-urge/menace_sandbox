import logging
import os
from tests.test_run_autonomous_env_vars import _load_module


def test_run_autonomous_adds_audit_handler(monkeypatch, tmp_path):
    log_path = tmp_path / "audit.log"
    monkeypatch.delenv("SANDBOX_CENTRAL_LOGGING", raising=False)
    monkeypatch.setenv("AUDIT_LOG_PATH", str(log_path))
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "validate_presets", lambda p: p)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *_a, **_k: True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "full_autonomous_run", lambda *a, **k: None)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    logging.getLogger().handlers.clear()
    mod.main(["--runs", "0", "--check-settings"])
    import logging_utils as lu
    assert os.getenv("SANDBOX_CENTRAL_LOGGING") == "1"
    assert any(isinstance(h, lu.AuditTrailHandler) for h in logging.getLogger().handlers)
