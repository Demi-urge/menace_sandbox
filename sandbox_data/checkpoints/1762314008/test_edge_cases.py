import json
import os
import importlib.util
from pathlib import Path

base = Path(__file__).resolve().parent / ".." / "sandbox_runner"
base = base.resolve()

spec = importlib.util.spec_from_file_location(
    "sandbox_runner.edge_case_generator", base / "edge_case_generator.py"  # path-ignore
)
edge_case_generator = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(edge_case_generator)  # type: ignore
generate_edge_cases = edge_case_generator.generate_edge_cases

spec2 = importlib.util.spec_from_file_location(
    "sandbox_runner.edge_case_plugin", base / "edge_case_plugin.py"  # path-ignore
)
edge_case_plugin = importlib.util.module_from_spec(spec2)  # type: ignore
spec2.loader.exec_module(edge_case_plugin)  # type: ignore
_load_edge_cases = edge_case_plugin._load_edge_cases


def test_generate_edge_cases_defaults():
    cases = generate_edge_cases()
    assert {
        "malformed.json",
        "timeout",
        "empty.txt",
        "null.txt",
        "invalid.bin",
        "http://edge-case.test/malformed",
        "http://edge-case.test/timeout",
        "http://edge-case.test/empty",
        "http://edge-case.test/null",
        "http://edge-case.test/invalid",
    }.issubset(cases.keys())


def test_generate_edge_cases_env(monkeypatch, tmp_path):
    extra = {"extra.json": "{}"}
    monkeypatch.setenv("SANDBOX_HOSTILE_PAYLOADS", json.dumps(extra))
    cases = generate_edge_cases()
    assert "extra.json" in cases
    monkeypatch.delenv("SANDBOX_HOSTILE_PAYLOADS", raising=False)


def test_edge_case_plugin_env(monkeypatch):
    payloads = {"x": "y"}
    monkeypatch.setenv("SANDBOX_EDGE_CASES", json.dumps(payloads))
    assert _load_edge_cases() == payloads
    monkeypatch.delenv("SANDBOX_EDGE_CASES", raising=False)
