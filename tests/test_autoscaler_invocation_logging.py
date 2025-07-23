import logging
import time

import menace.autoscaler as a
import menace.invocation_tracker as it


def test_autoscaler_logs_provider_errors(caplog):
    class BadProvider(a.Provider):
        def scale_up(self, amount: int = 1) -> bool:
            raise RuntimeError("fail up")

        def scale_down(self, amount: int = 1) -> bool:
            raise RuntimeError("fail down")

    auto = a.Autoscaler(provider=BadProvider(), cooldown=0)
    caplog.set_level(logging.ERROR)
    auto.scale_up()
    auto.scale_down()
    assert "provider scale_up failed" in caplog.text
    assert "provider scale_down failed" in caplog.text


def test_invocation_tracker_logs_errors(monkeypatch, caplog):
    monkeypatch.setattr(
        "builtins.open", lambda *a, **k: (_ for _ in ()).throw(IOError("fail"))
    )
    caplog.set_level(logging.ERROR)
    it.log_invocation(time.time(), "A")
    assert "log_invocation failed" in caplog.text
