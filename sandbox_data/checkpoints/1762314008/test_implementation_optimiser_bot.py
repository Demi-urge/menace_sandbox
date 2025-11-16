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
