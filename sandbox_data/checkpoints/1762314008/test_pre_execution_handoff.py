import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.pre_execution_roi_bot as prb  # noqa: E402
import menace.implementation_optimiser_bot as iob  # noqa: E402
import menace.task_handoff_bot as thb  # noqa: E402
import types  # noqa: E402


def test_handoff_to_implementation(monkeypatch):
    bot = prb.PreExecutionROIBot()
    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)
    optimiser = iob.ImplementationOptimiserBot(context_builder=builder)
    tasks = [
        prb.BuildTask(
            name="a",
            complexity=1.0,
            frequency=1.0,
            expected_income=1.0,
            resources={},
        )
    ]
    package = thb.TaskPackage(tasks=[])

    def fake_compile(infos):
        return package

    called = {"store": False, "send": False}

    def fake_store(infos, **kw):
        called["store"] = True
        return [1]

    def fake_send(pkg):
        called["send"] = True

    bot.handoff.compile = fake_compile
    bot.handoff.store_plan = fake_store
    bot.handoff.send_package = fake_send

    res = bot.handoff_to_implementation(tasks, optimiser, title="t")
    assert res is package
    assert called["store"] and called["send"]
    assert optimiser.history and optimiser.history[0] is package
