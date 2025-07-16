import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.bot_planning_bot as bpb


def _tasks():
    return [
        bpb.PlanningTask(
            description="collect data",
            complexity=1,
            frequency=1,
            expected_time=2.0,
            actions=["python"],
            resources={"cpu": 1},
        ),
        bpb.PlanningTask(
            description="analyse data",
            complexity=2,
            frequency=1,
            expected_time=3.0,
            actions=["python"],
            resources={"cpu": 2},
        ),
    ]


def test_evaluate_tasks():
    bot = bpb.BotPlanningBot()
    preds = bot.evaluate_tasks(_tasks())
    assert len(preds) == 2


def test_optimise_resources():
    bot = bpb.BotPlanningBot()
    alloc = bot.optimise_resources(_tasks(), cpu_limit=2)
    assert sum(a * t.resources.get("cpu", 1) for a, t in zip(alloc, _tasks())) <= 2 + 1e-5


def test_plan_bots_creates_graph():
    bot = bpb.BotPlanningBot()
    plans = bot.plan_bots(_tasks())
    assert plans
    assert bot.graph.number_of_nodes() == len(plans)
    assert plans[0].level.startswith("L") or plans[0].level.startswith("M")

