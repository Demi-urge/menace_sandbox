import sys
import types


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    pkg, _, sub = name.partition(".")
    if sub:
        pkg_mod = sys.modules.setdefault(pkg, types.ModuleType(pkg))
        setattr(pkg_mod, sub, mod)
    return mod

# minimal stubs for heavy dependencies of contrarian_model_bot
_stub(
    "menace.research_aggregator_bot",
    ResearchAggregatorBot=object,
    InfoDB=object,
    ResearchItem=object,
)
_stub("menace.chatgpt_enhancement_bot", EnhancementDB=object)
_stub("menace.prediction_manager_bot", PredictionManager=object)
_stub("menace.data_bot", DataBot=object)
_stub(
    "menace.strategy_prediction_bot",
    StrategyPredictionBot=object,
    CompetitorFeatures=object,
)
_stub("menace.resource_allocation_bot", ResourceAllocationBot=object)
_stub("menace.resource_prediction_bot", ResourceMetrics=object)
_stub("menace.task_handoff_bot", WorkflowDB=object, WorkflowRecord=object)
_stub("menace.contrarian_db", ContrarianDB=object, ContrarianRecord=object)
_stub("menace.capital_management_bot", CapitalManagementBot=object)
_stub("menace.unified_event_bus", UnifiedEventBus=object)

import menace.contrarian_model_bot as cmb


def test_workflow_to_dict_from_dict_roundtrip():
    wf = cmb.Workflow(
        name="demo",
        steps=[cmb.WorkflowStep(name="step", description="do", risk=0.1)],
        tags=["t1"],
    )
    data = wf.to_dict()
    restored = cmb.Workflow.from_dict(data)
    assert restored == wf
