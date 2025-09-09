import types

import meta_workflow_planner as mwp
import meta_workflow_planner_cli as cli


class SigGraph:
    def get_io_signature(self, wid):
        return {"inputs": {"x": "text/plain"}, "outputs": {"x": "text/plain"}}


class DummySynergyComparator:
    @staticmethod
    def compare(_a, _b):
        return types.SimpleNamespace(aggregate=0.0)


class DummyBuilder:
    def build(self, *_, **__):
        return {}

    def refresh_db_weights(self) -> None:
        pass


def test_simulate_multi_domain_chain(monkeypatch, capsys):
    monkeypatch.setattr(mwp, "ROITracker", None)
    monkeypatch.setattr(mwp, "WorkflowStabilityDB", None)
    monkeypatch.setattr(mwp, "WorkflowSynergyComparator", DummySynergyComparator)

    embeddings = {
        "YouTube": [1.0, 0.0],
        "YouTube2": [0.9, 0.435889894],
        "Reddit": [0.8, 0.6],
        "Email": [0.7, 0.714142842854285],
    }
    workflows = {
        "YouTube": {"domain": "youtube"},
        "YouTube2": {"domain": "youtube"},
        "Reddit": {"domain": "reddit"},
        "Email": {"domain": "email"},
    }

    def fake_encode(self, wid, _spec):
        return embeddings[wid]

    monkeypatch.setattr(mwp.MetaWorkflowPlanner, "encode_workflow", fake_encode)

    def fake_find_synergy_chain(start, length=5, context_builder=None):
        planner = mwp.MetaWorkflowPlanner(
            graph=SigGraph(), context_builder=context_builder
        )
        return planner.compose_pipeline(
            start, workflows, length=length, context_builder=context_builder
        )

    monkeypatch.setattr(cli, "find_synergy_chain", fake_find_synergy_chain)
    monkeypatch.setattr(cli, "_CONTEXT_BUILDER", DummyBuilder())

    args = types.SimpleNamespace(start="YouTube", length=3)
    cli._cmd_simulate(args)
    out = capsys.readouterr().out.strip()
    assert out == "YouTube -> YouTube2 -> Reddit"
