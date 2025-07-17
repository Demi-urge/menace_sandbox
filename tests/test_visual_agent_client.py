import importlib
import os


pkg_path = os.path.join(os.path.dirname(__file__), "..")
pkg_spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(pkg_path, "__init__.py"), submodule_search_locations=[pkg_path]
)
menace_pkg = importlib.util.module_from_spec(pkg_spec)
import sys
sys.modules["menace"] = menace_pkg
pkg_spec.loader.exec_module(menace_pkg)


def _reload_client(monkeypatch, prefix=None):
    if prefix is None:
        monkeypatch.delenv("VA_MESSAGE_PREFIX", raising=False)
    else:
        monkeypatch.setenv("VA_MESSAGE_PREFIX", prefix)
    mod = importlib.reload(importlib.import_module("menace.visual_agent_client"))
    return mod


def _capture_prompt(monkeypatch, vac_mod):
    captured = {}

    def fake_send(self, base, prompt):
        captured["prompt"] = prompt
        return True, "ok"

    monkeypatch.setattr(vac_mod.VisualAgentClient, "_send", fake_send)
    client = vac_mod.VisualAgentClient(urls=["http://x"])
    client.ask([{"content": "hello"}])
    return captured["prompt"], vac_mod.SELF_IMPROVEMENT_PREFIX


def test_default_prefix(monkeypatch):
    vac_mod = _reload_client(monkeypatch)
    prompt, prefix = _capture_prompt(monkeypatch, vac_mod)
    assert prompt.startswith(prefix)
    assert prefix == vac_mod.DEFAULT_MESSAGE_PREFIX


def test_env_prefix(monkeypatch):
    vac_mod = _reload_client(monkeypatch, "CUSTOM")
    prompt, _ = _capture_prompt(monkeypatch, vac_mod)
    assert prompt.startswith("CUSTOM")

