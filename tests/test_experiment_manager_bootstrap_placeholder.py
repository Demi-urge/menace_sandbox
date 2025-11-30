import importlib
import os
import sys
import types

import pytest


os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

vs = types.ModuleType("vector_service")


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *args, **kwargs):
        return {}


vs.ContextBuilder = DummyBuilder
sys.modules.setdefault("vector_service", vs)


def test_reuses_broker_placeholder(monkeypatch):
    placeholder_pipeline = types.SimpleNamespace(
        context_builder=DummyBuilder(), bootstrap_placeholder=True
    )
    placeholder_manager = object()

    class DummyBroker:
        def __init__(self):
            self.advertisements: list[dict[str, object | None]] = []

        def advertise(
            self,
            *,
            pipeline: object | None = None,
            sentinel: object | None = None,
            owner: bool | None = None,
        ) -> None:
            self.advertisements.append(
                {"pipeline": pipeline, "sentinel": sentinel, "owner": owner}
            )

        def resolve(self) -> tuple[object | None, object | None]:
            return placeholder_pipeline, placeholder_manager

    broker = DummyBroker()

    monkeypatch.setattr(
        "menace.bootstrap_placeholder.advertise_broker_placeholder",
        lambda: (placeholder_pipeline, placeholder_manager, broker),
    )

    exp_mod = importlib.import_module("menace.experiment_manager")
    monkeypatch.setattr(exp_mod, "_BOOTSTRAP_PLACEHOLDER", placeholder_pipeline, raising=False)
    monkeypatch.setattr(exp_mod, "_BOOTSTRAP_SENTINEL", placeholder_manager, raising=False)
    monkeypatch.setattr(exp_mod, "_BOOTSTRAP_BROKER", broker, raising=False)

    claim_kwargs: list[dict[str, object]] = []

    def fake_claim_bootstrap_dependency_entry(**kwargs):
        claim_kwargs.append(kwargs)
        broker.advertise(
            pipeline=kwargs.get("pipeline"),
            sentinel=kwargs.get("manager"),
            owner=kwargs.get("owner"),
        )

        def _promote(manager):
            placeholder_pipeline.manager = manager
            broker.advertise(
                pipeline=placeholder_pipeline, sentinel=manager, owner=True
            )

        return placeholder_pipeline, _promote, placeholder_manager, False

    monkeypatch.setattr(
        exp_mod, "claim_bootstrap_dependency_entry", fake_claim_bootstrap_dependency_entry
    )

    def fake_prepare_pipeline_for_bootstrap(**_kwargs):
        pytest.fail("prepare_pipeline_for_bootstrap should not run when placeholder is active")

    monkeypatch.setattr(
        exp_mod, "prepare_pipeline_for_bootstrap", fake_prepare_pipeline_for_bootstrap
    )

    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=1: types.SimpleNamespace(empty=True))
    )
    capital_bot = types.SimpleNamespace()
    builder = DummyBuilder()

    mgr = exp_mod.ExperimentManager(data_bot, capital_bot, context_builder=builder)

    assert mgr.pipeline is placeholder_pipeline
    assert claim_kwargs and claim_kwargs[0]["owner"] is True
    assert broker.advertisements[0]["pipeline"] is placeholder_pipeline

    mgr.promote_pipeline_manager("real-manager")

    assert broker.advertisements[-1] == {
        "pipeline": placeholder_pipeline,
        "sentinel": "real-manager",
        "owner": True,
    }
