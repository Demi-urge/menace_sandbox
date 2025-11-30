import threading

import coding_bot_interface as cbi
import pytest


class DummyPromise:
    def __init__(self):
        self.waiters = 0
        self.wait_called = 0
        self.result = ("active-pipeline", self._promote)

    def _promote(self, *_args):
        return None

    def wait(self):
        self.wait_called += 1
        return self.result


class DummyCoordinator:
    def __init__(self, promise):
        self._promise = promise

    def peek_active(self):
        return self._promise


class DummyBroker:
    def __init__(
        self, pipeline="broker-pipeline", sentinel="broker-sentinel", owner_active: bool = True
    ):
        self.pipeline = pipeline
        self.sentinel = sentinel
        self.advertise_calls = 0
        self.active_owner = owner_active

    def resolve(self):
        return self.pipeline, self.sentinel

    def advertise(self, *args, **kwargs):  # pragma: no cover - should not be invoked
        self.advertise_calls += 1
        raise AssertionError("advertise should not be called during recursion guard")


def test_recursion_guard_returns_active_promise(monkeypatch, caplog):
    cbi._REENTRY_ATTEMPTS.clear()
    promise = DummyPromise()
    broker = DummyBroker()

    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", DummyCoordinator(promise))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(cbi._BOOTSTRAP_STATE, "depth", 1)
    caplog.set_level("INFO")

    try:
        result = cbi.prepare_pipeline_for_bootstrap(
            pipeline_cls=type("Pipeline", (), {}),
            context_builder=None,
            bot_registry=None,
            data_bot=None,
        )
    finally:
        try:
            delattr(cbi._BOOTSTRAP_STATE, "depth")
        except AttributeError:
            pass

    assert result == promise.result
    assert promise.waiters == 1
    assert promise.wait_called == 1
    assert broker.advertise_calls == 0
    assert any(
        "recursion_guard_promise_short_circuit" in record.message for record in caplog.records
    )


def test_recursion_guard_reuses_broker_without_advertise(monkeypatch, caplog):
    cbi._REENTRY_ATTEMPTS.clear()
    broker = DummyBroker()

    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", DummyCoordinator(None))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(cbi._BOOTSTRAP_STATE, "depth", 1)
    caplog.set_level("INFO")

    try:
        pipeline, promote = cbi.prepare_pipeline_for_bootstrap(
            pipeline_cls=type("Pipeline", (), {}),
            context_builder=None,
            bot_registry=None,
            data_bot=None,
        )
    finally:
        try:
            delattr(cbi._BOOTSTRAP_STATE, "depth")
        except AttributeError:
            pass

    assert pipeline == broker.pipeline
    promote("real-manager")
    assert broker.advertise_calls == 0
    assert any(
        "recursion_guard_broker_short_circuit" in record.message for record in caplog.records
    )


def test_recursion_guard_short_circuits_concurrent_callers(monkeypatch):
    cbi._REENTRY_ATTEMPTS.clear()
    promise = DummyPromise()
    broker = DummyBroker()
    results: list[tuple] = []

    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", DummyCoordinator(promise))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)

    def _invoke():
        setattr(cbi._BOOTSTRAP_STATE, "depth", 1)
        try:
            results.append(
                cbi.prepare_pipeline_for_bootstrap(
                    pipeline_cls=type("Pipeline", (), {}),
                    context_builder=None,
                    bot_registry=None,
                    data_bot=None,
                )
            )
        finally:
            try:
                delattr(cbi._BOOTSTRAP_STATE, "depth")
            except AttributeError:
                pass

    threads = [threading.Thread(target=_invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [promise.result, promise.result]
    assert promise.waiters == 2
    assert promise.wait_called == 2
    assert broker.advertise_calls == 0

    try:
        delattr(cbi._BOOTSTRAP_STATE, "depth")
    except AttributeError:
        pass


def test_recursion_guard_blocks_orphaned_broker_placeholder(monkeypatch, caplog):
    cbi._REENTRY_ATTEMPTS.clear()
    broker = DummyBroker(owner_active=False)

    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", DummyCoordinator(None))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(cbi._BOOTSTRAP_STATE, "depth", 1)
    caplog.set_level("INFO")

    try:
        with pytest.raises(RuntimeError):
            cbi.prepare_pipeline_for_bootstrap(
                pipeline_cls=type("Pipeline", (), {}),
                context_builder=None,
                bot_registry=None,
                data_bot=None,
            )
    finally:
        try:
            delattr(cbi._BOOTSTRAP_STATE, "depth")
        except AttributeError:
            pass

    assert broker.advertise_calls == 0
    assert any(
        "recursion_guard_placeholder_block" in record.message for record in caplog.records
    )


def test_recursion_guard_enforces_reentry_cap(monkeypatch, caplog):
    cbi._REENTRY_ATTEMPTS.clear()
    caplog.set_level("INFO")
    promise = DummyPromise()
    broker = DummyBroker()

    monkeypatch.setenv("MENACE_BOOTSTRAP_REENTRY_CAP", "1")
    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", DummyCoordinator(promise))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(cbi._BOOTSTRAP_STATE, "depth", 1)

    try:
        cbi.prepare_pipeline_for_bootstrap(
            pipeline_cls=type("Pipeline", (), {}),
            context_builder=None,
            bot_registry=None,
            data_bot=None,
        )

        with pytest.raises(RuntimeError):
            cbi.prepare_pipeline_for_bootstrap(
                pipeline_cls=type("Pipeline", (), {}),
                context_builder=None,
                bot_registry=None,
                data_bot=None,
            )
    finally:
        try:
            delattr(cbi._BOOTSTRAP_STATE, "depth")
        except AttributeError:
            pass

    assert promise.waiters == 1
    assert any("reentry_cap_exceeded" in record.message for record in caplog.records)
