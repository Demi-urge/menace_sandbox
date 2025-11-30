import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.task_handoff_bot as thb  # noqa: E402
import menace.implementation_optimiser_bot as iob  # noqa: E402
import types  # noqa: E402


def test_process_records_package():
    pkg = thb.TaskPackage(tasks=[
        thb.TaskInfo(
            name="t",
            dependencies=[],
            resources={},
            schedule="once",
            code="print('x')",
            metadata={},
        )
    ])
    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)
    bot = iob.ImplementationOptimiserBot(context_builder=builder)
    advice = bot.process(pkg)
    assert bot.history and bot.history[0] is pkg
    assert advice[0].name == "t"


def test_refresh_db_weights_failure():
    class BadBuilder:
        def refresh_db_weights(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        iob.ImplementationOptimiserBot(context_builder=BadBuilder())


def test_reuses_active_bootstrap_promise(monkeypatch):
    dependency_broker = iob._bootstrap_dependency_broker()
    dependency_broker.clear()
    iob._pipeline_promoter = None

    sentinel_manager = types.SimpleNamespace()
    promise_pipeline = types.SimpleNamespace(manager=sentinel_manager)
    promote_calls: list[types.SimpleNamespace | None] = []
    prepare_called: list[str] = []

    def _promote(manager):
        promote_calls.append(manager)

    class _Promise:
        waiters = 1

        def wait(self):
            return promise_pipeline, _promote

    def _prepare(**_kwargs):  # pragma: no cover - should not be invoked
        prepare_called.append("prepare")
        return None, lambda *_args, **_kwargs: None

    monkeypatch.setattr(iob, "prepare_pipeline_for_bootstrap", _prepare)
    monkeypatch.setattr(
        iob,
        "_GLOBAL_BOOTSTRAP_COORDINATOR",
        types.SimpleNamespace(peek_active=lambda: _Promise()),
    )
    monkeypatch.setattr(iob, "_resolve_pipeline_cls", lambda: type("Dummy", (), {}))
    monkeypatch.setattr(iob, "get_active_bootstrap_pipeline", lambda: (None, None))
    monkeypatch.setattr(iob, "_current_bootstrap_context", lambda: None)

    pipeline_factory = iob._bootstrap_pipeline_factory()
    pipeline = pipeline_factory(types.SimpleNamespace())

    assert pipeline is promise_pipeline
    assert not prepare_called
    assert dependency_broker.resolve()[0] is pipeline
    assert iob._pipeline_promoter is not None

    iob._pipeline_promoter(sentinel_manager)
    assert promote_calls == [sentinel_manager]
