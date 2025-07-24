import logging
import os

import menace.visual_agent_client as vac


def test_release_logs_errors(monkeypatch, caplog, tmp_path):
    caplog.set_level(logging.WARNING)

    monkeypatch.setattr(vac._ContextFileLock, "__del__", lambda self: None)
    monkeypatch.setattr(vac._global_lock, "__del__", lambda self: None, raising=False)
    vac._global_lock.release(force=True)
    monkeypatch.setattr(vac, "_global_lock", vac._ContextFileLock(str(tmp_path / "gl")))
    client = vac._ContextFileLock(str(tmp_path / "lock"))

    def fake_remove(path):
        raise FileNotFoundError

    monkeypatch.setattr(os, "remove", fake_remove)
    client.release()
    assert any("lock file already removed" in rec.message for rec in caplog.records)

    def fake_remove2(path):
        raise OSError("boom")

    monkeypatch.setattr(os, "remove", fake_remove2)
    client.release()
    assert any("failed to remove lock file" in rec.message for rec in caplog.records)


