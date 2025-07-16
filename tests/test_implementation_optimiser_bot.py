import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.task_handoff_bot as thb
import menace.implementation_optimiser_bot as iob


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
    bot = iob.ImplementationOptimiserBot()
    advice = bot.process(pkg)
    assert bot.history and bot.history[0] is pkg
    assert advice[0].name == "t"
