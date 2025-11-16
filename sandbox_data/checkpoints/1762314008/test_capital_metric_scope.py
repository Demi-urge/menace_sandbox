import os
import sys
import types
import asyncio
from menace.db_router import init_db_router

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# Stub optional dependencies to keep tests lightweight
stub = types.ModuleType("jinja2")
stub.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", stub)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))

stub.__spec__ = types.SimpleNamespace()
sys.modules.setdefault("jinja2.ext", types.ModuleType("ext"))

# Stub transformers to avoid heavy dependencies during import
trans_mod = types.ModuleType("transformers")
trans_mod.AutoModel = object
trans_mod.AutoTokenizer = object
sys.modules.setdefault("transformers", trans_mod)

import menace.capital_management_bot as cmb  # noqa: E402


def _setup_router(tmp_path):
    router = init_db_router("scope", str(tmp_path / "loc.db"), str(tmp_path / "sha.db"))
    conn = router.get_connection("metrics")
    conn.execute(
        "CREATE TABLE metrics(name TEXT, value REAL, ts TEXT, source_menace_id TEXT)"
    )
    conn.execute(
        "INSERT INTO metrics VALUES('m', 1.0, '2024-01-01', ?)",
        (router.menace_id,),
    )
    conn.execute(
        "INSERT INTO metrics VALUES('m', 2.0, '2024-01-02', 'other')"
    )
    conn.commit()
    return router


def test_fetch_metric_scopes(tmp_path):
    router = _setup_router(tmp_path)
    assert cmb.fetch_metric_from_db("m", router=router, scope="local") == 1.0
    assert cmb.fetch_metric_from_db("m", router=router, scope="global") == 2.0
    # All should return the latest entry regardless of menace
    assert cmb.fetch_metric_from_db("m", router=router, scope="all") == 2.0


def test_async_metric_scopes(tmp_path, monkeypatch):
    router = _setup_router(tmp_path)
    monkeypatch.setattr(cmb, "GLOBAL_ROUTER", router)
    res = asyncio.run(
        cmb.fetch_capital_metrics_async(metric_names=["m"], scope="local")
    )
    assert isinstance(res, cmb.CapitalMetrics)
    assert res["m"] == 1.0
    res = asyncio.run(
        cmb.fetch_capital_metrics_async(metric_names=["m"], scope="global")
    )
    assert res["m"] == 2.0
    res = asyncio.run(
        cmb.fetch_capital_metrics_async(metric_names=["m"], scope="all")
    )
    assert res["m"] == 2.0
