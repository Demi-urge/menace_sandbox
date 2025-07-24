import logging
from pathlib import Path
import types
import builtins

import menace.models_repo as mrepo


def test_model_build_lock_logs_exception(monkeypatch, caplog, tmp_path):
    caplog.set_level(logging.ERROR)

    monkeypatch.setattr(mrepo, "MODELS_REPO_PATH", tmp_path)
    monkeypatch.setattr(mrepo, "ACTIVE_MODEL_FILE", tmp_path / ".active_model")

    def fake_unlink(self):
        raise RuntimeError("fail")

    monkeypatch.setattr(Path, "unlink", fake_unlink)

    with mrepo.model_build_lock(1):
        pass

    assert any("failed to remove active model file" in rec.message for rec in caplog.records)

