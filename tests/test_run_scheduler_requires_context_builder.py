import sys
import types
import importlib
import pytest

# Stub modules to avoid heavy dependencies during import
wc_suggester = types.ModuleType('workflow_chain_suggester')
class WorkflowChainSuggester:
    pass
wc_suggester.WorkflowChainSuggester = WorkflowChainSuggester
sys.modules['workflow_chain_suggester'] = wc_suggester

we_manager = types.ModuleType('workflow_evolution_manager')
we_manager._build_callable = lambda seq: None
sys.modules['workflow_evolution_manager'] = we_manager

cws = types.ModuleType('composite_workflow_scorer')
class CompositeWorkflowScorer:
    pass
cws.CompositeWorkflowScorer = CompositeWorkflowScorer
sys.modules['composite_workflow_scorer'] = cws

wsc = types.ModuleType('workflow_synergy_comparator')
class WorkflowSynergyComparator:
    @staticmethod
    def _entropy(spec):  # pragma: no cover - not used
        return 0.0
wsc.WorkflowSynergyComparator = WorkflowSynergyComparator
sys.modules['workflow_synergy_comparator'] = wsc

wsd = types.ModuleType('workflow_stability_db')
class WorkflowStabilityDB:
    pass
wsd.WorkflowStabilityDB = WorkflowStabilityDB
sys.modules['workflow_stability_db'] = wsd

meta_mod = types.ModuleType('meta_workflow_planner')
class MetaWorkflowPlanner:
    def __init__(self, *a, **k):
        pass
    def schedule(self, *a, **k):  # pragma: no cover - not used
        return []
meta_mod.MetaWorkflowPlanner = MetaWorkflowPlanner
meta_mod.simulate_meta_workflow = lambda *a, **k: {}
sys.modules['meta_workflow_planner'] = meta_mod

sys.modules['workflow_run_summary'] = types.ModuleType('workflow_run_summary')

# Import the module under test after stubbing dependencies
workflow_chain_simulator = importlib.import_module('workflow_chain_simulator')


def test_run_scheduler_requires_context_builder():
    with pytest.raises(TypeError):
        workflow_chain_simulator.run_scheduler({})
