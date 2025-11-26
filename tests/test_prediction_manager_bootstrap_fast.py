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
