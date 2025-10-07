from __future__ import annotations

import types

import self_coding_dependency_probe as probe


def test_probe_reports_missing(monkeypatch):
    checked = {}

    def fake_find_spec(name: str):
        checked[name] = checked.get(name, 0) + 1
        if name == "pydantic":
            return None
        return types.SimpleNamespace()

    monkeypatch.setattr(probe, "find_spec", fake_find_spec)

    ready, missing = probe.ensure_self_coding_ready()

    assert not ready
    assert "pydantic" in missing
    assert checked["pydantic"] == 1


def test_probe_all_present(monkeypatch):
    monkeypatch.setattr(
        probe,
        "find_spec",
        lambda name: types.SimpleNamespace(),
    )

    ready, missing = probe.ensure_self_coding_ready(["pydantic", "sklearn"])

    assert ready
    assert missing == ()
