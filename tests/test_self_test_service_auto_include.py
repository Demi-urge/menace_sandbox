import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "menace.self_test_service",
    ROOT / "self_test_service.py",
)
mod = importlib.util.module_from_spec(spec)

pkg = sys.modules.get("menace")
if pkg is not None:
    pkg.__path__ = [str(ROOT)]

spec.loader.exec_module(mod)


def test_auto_include_isolated_env(monkeypatch):
    monkeypatch.setenv("SANDBOX_AUTO_INCLUDE_ISOLATED", "1")
    svc = mod.SelfTestService(discover_isolated=False, recursive_isolated=False)
    assert svc.discover_isolated is True
    assert svc.recursive_isolated is True
