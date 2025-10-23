"""Regression tests for recursive YAML fallback handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from menace_sandbox import self_coding_thresholds as sct


def test_load_config_recursion_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ensure recursive load errors fall back to the lightweight parser."""

    cfg = tmp_path / "self_coding_thresholds.yaml"
    cfg.write_text("default: {}\n", encoding="utf-8")

    def _boom(_: str):
        raise RecursionError("recursive anchor detected")

    fallback_result = {"default": {"roi_drop": -0.25}}

    monkeypatch.setattr(sct.yaml, "safe_load", _boom)
    monkeypatch.setattr(sct._FALLBACK_YAML, "safe_load", lambda text: fallback_result)

    data = sct._load_config(cfg)
    assert data == fallback_result


def test_recursion_switches_future_loads_to_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "self_coding_thresholds.yaml"
    cfg.write_text("default: {}\n", encoding="utf-8")

    monkeypatch.setattr(sct, "_FORCED_FALLBACK_PATHS", set())

    calls: list[str] = []

    def _boom(_: str):
        calls.append("pyyaml")
        raise RecursionError("recursive anchor detected")

    fallback_result = {"default": {"roi_drop": -0.25}}
    monkeypatch.setattr(sct.yaml, "safe_load", _boom)
    monkeypatch.setattr(sct._FALLBACK_YAML, "safe_load", lambda text: fallback_result)

    assert sct._load_config(cfg) == fallback_result
    assert calls == ["pyyaml"]

    # Subsequent loads should bypass the primary loader entirely once recursion was detected.
    monkeypatch.setattr(sct._FALLBACK_YAML, "safe_load", lambda text: {"default": {"roi_drop": -0.33}})
    monkeypatch.setattr(sct.yaml, "safe_load", lambda text: (_ for _ in ()).throw(RuntimeError("should not be called")))

    data = sct._load_config(cfg)
    assert data["default"]["roi_drop"] == -0.33


def test_dump_config_recursion_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ensure recursion during dumping switches to the fallback serializer."""

    cfg = tmp_path / "self_coding_thresholds.yaml"

    def _boom(*_args, **_kwargs):
        raise RecursionError("cannot serialise recursive structure")

    rendered = "default: {}\n"

    monkeypatch.setattr(sct.yaml, "safe_dump", _boom)
    monkeypatch.setattr(sct._FALLBACK_YAML, "safe_dump", lambda data, sort_keys=False: rendered)

    sct.update_thresholds("bot-a", path=cfg)

    assert cfg.read_text(encoding="utf-8") == rendered
