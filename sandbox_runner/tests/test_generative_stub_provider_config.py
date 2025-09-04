import json
import sys
import types
from pathlib import Path
from dynamic_path_router import resolve_path

sys.modules.setdefault(
    "db_router", types.SimpleNamespace(GLOBAL_ROUTER=None, init_db_router=lambda *a, **k: None)
)

sys.path.append(str(resolve_path("")))

from sandbox_runner import generative_stub_provider as gsp  # noqa: E402


def test_config_reload(monkeypatch):
    monkeypatch.setenv("SANDBOX_STUB_TIMEOUT", "1")
    monkeypatch.setenv("SANDBOX_STUB_MAX_CONCURRENCY", "1")
    cfg1 = gsp.get_config(refresh=True)
    assert cfg1.timeout == 1.0
    sem1 = cfg1.rate_limit

    monkeypatch.setenv("SANDBOX_STUB_TIMEOUT", "2")
    monkeypatch.setenv("SANDBOX_STUB_MAX_CONCURRENCY", "2")
    cfg2 = gsp.get_config(refresh=True)
    assert cfg2.timeout == 2.0
    assert cfg2.max_concurrency == 2
    # semaphore replaced
    assert sem1 is not cfg2.rate_limit


def test_save_cache_validates_entries(tmp_path):
    cfg = gsp.get_config(refresh=True)
    cfg.cache_path = tmp_path / "cache.json"
    with gsp._CACHE_LOCK:
        gsp._CACHE.clear()
        gsp._CACHE[("ok", "func")] = {"a": 1}
        gsp._CACHE[("bad", "func")] = {"a": object()}
    gsp._save_cache(cfg)
    data = json.loads((tmp_path / "cache.json").read_text())
    assert data == [["ok::func", {"a": 1}]]
