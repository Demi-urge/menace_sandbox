import importlib
import sys
import types

from vector_service import registry

PACKAGE_PREFIX = "menace_sandbox"

# Provide stub run_autonomous module to avoid heavy dependency checks
_stub = types.ModuleType("run_autonomous")
_stub.LOCAL_KNOWLEDGE_MODULE = None
sys.modules.setdefault("run_autonomous", _stub)
sys.modules.setdefault(f"{PACKAGE_PREFIX}.run_autonomous", _stub)

def _import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return importlib.import_module(f"{PACKAGE_PREFIX}.{name}")


def test_vector_registry_modalities():
    assert registry._VECTOR_REGISTRY, "Vector registry is empty"

    for kind, (vec_mod, vec_cls, db_mod, db_cls) in registry._VECTOR_REGISTRY.items():
        vec_module = _import(vec_mod)
        vec_cls_obj = getattr(vec_module, vec_cls)
        vec_instance = vec_cls_obj()
        vec = vec_instance.transform({})
        assert isinstance(vec, list) and vec, f"{kind} vectorizer returned invalid vector"

        if db_mod and db_cls:
            db_module = _import(db_mod)
            db_class = getattr(db_module, db_cls)
            assert hasattr(db_class, "backfill_embeddings"), (
                f"{kind} database class {db_cls} lacks backfill_embeddings"
            )
