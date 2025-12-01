import threading
from types import SimpleNamespace

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
    setattr(cbi._BOOTSTRAP_STATE, "depth", 1)
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
    setattr(cbi._BOOTSTRAP_STATE, "depth", 1)
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


def test_recursion_guard_seeds_orphaned_broker_placeholder(monkeypatch, caplog):
    cbi._REENTRY_ATTEMPTS.clear()

    class RecordingBroker(DummyBroker):
        def __init__(self):
            sentinel = SimpleNamespace()
            cbi._mark_bootstrap_placeholder(sentinel)
            pipeline = cbi._build_bootstrap_placeholder_pipeline(sentinel)
            super().__init__(pipeline=pipeline, sentinel=sentinel, owner_active=False)
            self.calls: list[tuple[object, object, bool]] = []

        def advertise(self, pipeline=None, sentinel=None, owner=False):
            self.advertise_calls += 1
            if pipeline is not None:
                self.pipeline = pipeline
            if sentinel is not None:
                self.sentinel = sentinel
            if owner:
                self.active_owner = True
            self.calls.append((pipeline, sentinel, owner))
            return self.pipeline, self.sentinel

    broker = RecordingBroker()

    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", DummyCoordinator(None))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    setattr(cbi._BOOTSTRAP_STATE, "depth", 1)
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

    assert broker.advertise_calls == 1
    assert broker.calls[-1][2] is True
    assert pipeline == broker.pipeline
    promote("real-manager")
    assert any(
        "placeholder_broker_seeded" in record.message for record in caplog.records
    )


def test_recursion_guard_enforces_reentry_cap(monkeypatch, caplog):
    cbi._REENTRY_ATTEMPTS.clear()
    caplog.set_level("INFO")
    promise = DummyPromise()
    broker = DummyBroker()

    monkeypatch.setenv("MENACE_BOOTSTRAP_REENTRY_CAP", "1")
    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", DummyCoordinator(promise))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    setattr(cbi._BOOTSTRAP_STATE, "depth", 1)

    try:
        cbi.prepare_pipeline_for_bootstrap(
            pipeline_cls=type("Pipeline", (), {}),
            context_builder=None,
            bot_registry=None,
            data_bot=None,
        )
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

    assert promise.waiters == 2
    assert any("reentry_cap_exceeded" in record.message for record in caplog.records)
    assert any("reentry_cap_exceeded" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    "caller_module",
    [
        "research_aggregator_bot",
        "prediction_manager_bot",
        "cognition_layer",
        "menace_orchestrator",
    ],
)
def test_priority_callers_normalized_for_recursion_guard(monkeypatch, caplog, caller_module):
    cbi._REENTRY_ATTEMPTS.clear()
    promise = DummyPromise()
    broker = DummyBroker()

    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", DummyCoordinator(promise))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    setattr(cbi._BOOTSTRAP_STATE, "depth", 1)
    caplog.set_level("INFO")

    def _caller_details():
        return {"module": caller_module, "path": None, "stack_signature": "sig"}

    monkeypatch.setattr(cbi, "_resolve_caller_details", _caller_details)

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
    assert any("priority_promise_short_circuit" in record.message for record in caplog.records)
    assert any(
        getattr(record, "caller_module_normalized", None) in {
            "researchaggregator",
            "predictionmanager",
            "cognitionlayer",
            "menaceorchestrator",
        }
        for record in caplog.records
    )


def test_priority_callers_seed_broker_and_reuse_promise(monkeypatch, caplog):
    cbi._REENTRY_ATTEMPTS.clear()
    sentinel = SimpleNamespace()
    cbi._mark_bootstrap_placeholder(sentinel)
    placeholder_pipeline = cbi._build_bootstrap_placeholder_pipeline(sentinel)

    class PlaceholderBroker:
        def __init__(self):
            self.pipeline = placeholder_pipeline
            self.sentinel = sentinel
            self.active_owner = False
            self.calls: list[tuple[object, object, bool]] = []

        def resolve(self):
            return self.pipeline, self.sentinel

        def advertise(self, pipeline=None, sentinel=None, owner=False):
            if pipeline is not None:
                self.pipeline = pipeline
            if sentinel is not None:
                self.sentinel = sentinel
            if owner:
                self.active_owner = True
            self.calls.append((pipeline, sentinel, owner))
            return self.pipeline, self.sentinel

    class ActivePromise:
        def __init__(self, result):
            self.waiters = 0
            self._result = result
            self._event = threading.Event()
            self._event.set()

        def wait(self):
            self._event.wait()
            return self._result

    class ActiveCoordinator:
        def __init__(self, promise):
            self._promise = promise

        def peek_active(self):
            return self._promise

    promise_result = ("running-pipeline", lambda *_args: None)
    promise = ActivePromise(promise_result)
    broker = PlaceholderBroker()

    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", ActiveCoordinator(promise))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(cbi, "_prepare_pipeline_for_bootstrap_impl", lambda **_: (_ for _ in ()).throw(AssertionError("should not run bootstrap impl")))
    monkeypatch.setattr(
        cbi,
        "_resolve_caller_details",
        lambda: {"module": "menace_orchestrator", "path": None, "stack_signature": "sig"},
    )
    monkeypatch.setattr(cbi, "_resolve_caller_module_name", lambda: "menace_orchestrator")
    caplog.set_level("INFO")

    results: list[tuple[object, object]] = []

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

    threads = [threading.Thread(target=_invoke) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [promise_result, promise_result, promise_result]
    assert promise.waiters == 3
    assert broker.calls[0][2] is True
    assert broker.active_owner is True
    assert broker.calls.count(broker.calls[0]) == 1
    assert any("placeholder_promise_reuse" in record.message for record in caplog.records)
