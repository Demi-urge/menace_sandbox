import json
import types
import sys

plugins_stub = types.ModuleType("menace.plugins")
plugins_stub.load_plugins = lambda sub: None
sys.modules.setdefault("menace.plugins", plugins_stub)

db_router_stub = types.ModuleType("db_router")
db_router_stub.init_db_router = lambda *a, **k: None
sys.modules.setdefault("db_router", db_router_stub)

code_db_stub = types.ModuleType("code_database")
code_db_stub.PatchHistoryDB = object
sys.modules.setdefault("code_database", code_db_stub)

patch_prov_stub = types.ModuleType("patch_provenance")
patch_prov_stub.PatchLogger = object
patch_prov_stub.build_chain = lambda *a, **k: None
patch_prov_stub.search_patches_by_vector = lambda *a, **k: None
patch_prov_stub.search_patches_by_license = lambda *a, **k: None
sys.modules.setdefault("patch_provenance", patch_prov_stub)

cache_utils_stub = types.ModuleType("cache_utils")
cache_utils_stub.get_cached_chain = lambda *a, **k: None
cache_utils_stub.set_cached_chain = lambda *a, **k: None
cache_utils_stub._get_cache = lambda *a, **k: None
cache_utils_stub.clear_cache = lambda *a, **k: None
cache_utils_stub.show_cache = lambda *a, **k: {}
cache_utils_stub.cache_stats = lambda *a, **k: {}
sys.modules.setdefault("cache_utils", cache_utils_stub)

workflow_stub = types.ModuleType("workflow_synthesizer_cli")
workflow_stub.run = lambda *a, **k: 0
sys.modules.setdefault("workflow_synthesizer_cli", workflow_stub)

import menace_cli


def test_branch_log_cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(menace_cli, "load_plugins", lambda sub: None)
    log_path = tmp_path / "audit.log"
    entry = {"action": "patch_branch", "patch_id": "1", "branch": "review/1"}
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"- {json.dumps(entry)}\n")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(log_path))
    assert menace_cli.main(["branch-log"]) == 0
    out = capsys.readouterr().out.strip()
    assert "review/1" in out
