"""Bootstrap-fast regression tests for :mod:`prediction_manager_bot`."""

import types
import menace.prediction_manager_bot as pmb


class _FailingDataBot:
    def __getattr__(self, name: str):  # pragma: no cover - defensive
        raise AssertionError(f"DataBot should not be accessed during bootstrap_fast (got {name})")


class _StubDataBot:
    def __init__(self) -> None:
        self.db = types.SimpleNamespace(fetch=lambda _limit=20: [])


def test_bootstrap_fast_skips_default_registration(tmp_path, monkeypatch):
    """Fast bootstrap avoids spinning up default metric bots."""

    monkeypatch.setattr(pmb, "_import_future_prediction_bots", lambda: (_ for _ in ()).throw(AssertionError("futures")))
    manager = pmb.PredictionManager(
        tmp_path / "reg.json", data_bot=_FailingDataBot(), bootstrap_fast=True
    )
    assert manager.registry == {}
    assert manager.bootstrap_fast is True


def test_standard_bootstrap_populates_registry(tmp_path):
    """Normal initialisation still hydrates the default metric bots."""

    manager = pmb.PredictionManager(
        tmp_path / "reg.json", data_bot=_StubDataBot(), bootstrap_fast=False
    )
    registered_metrics = {metric for entry in manager.registry.values() for metric in entry.profile.get("metric", [])}
    assert registered_metrics, "default bots should register metrics when not bootstrapping fast"

def test_prediction_manager_reuses_placeholder_when_broker_inactive(monkeypatch):
    import importlib

    module = importlib.reload(importlib.import_module("menace_sandbox.prediction_manager_bot"))

    placeholder_pipeline = object()
    placeholder_manager = object()
    broker = types.SimpleNamespace(
        active_owner=False,
        active_pipeline=placeholder_pipeline,
        active_sentinel=placeholder_manager,
    )

    monkeypatch.setattr(
        module,
        "resolve_bootstrap_placeholders",
        lambda **_: (placeholder_pipeline, placeholder_manager, broker),
    )

    pipeline, manager, resolved_broker = module._bootstrap_placeholders()

    assert pipeline is placeholder_pipeline
    assert manager is placeholder_manager
    assert resolved_broker is broker
