from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

MODULES = [
    "embeddable_db_mixin",
    "gpt_memory",
    "shared_gpt_memory",
    "shared_knowledge_module",
    "local_knowledge_module",
    "data_bot",
    "coding_bot_interface",
    "quick_fix_engine",
]

OPTIONAL_DEPENDENCIES = {
    "annoy",
    "faiss",
    "faiss_cpu",
    "numpy",
    "sentence_transformers",
    "sklearn",
    "pylint",
}


@pytest.mark.parametrize("module_name", MODULES)
def test_flat_import_aliases(module_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))

    # Ensure a clean slate for the module and helper.
    for key in (
        module_name,
        f"menace_sandbox.{module_name}",
        "menace_sandbox.import_compat",
        "import_compat",
        "menace_sandbox",
    ):
        sys.modules.pop(key, None)

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in OPTIONAL_DEPENDENCIES:
            pytest.skip(f"optional dependency {exc.name} missing for {module_name}")
        raise
    except ImportError as exc:
        message = str(exc)
        if "Self-coding engine is required" in message or "self_coding_managed" in message:
            pytest.skip("self_coding_engine helpers required for coding bot tests")
        raise
    except RuntimeError as exc:
        if "context_builder_util" in str(exc):
            pytest.skip("context_builder_util helpers required for quick_fix_engine")
        raise

    qualified = f"menace_sandbox.{module_name}"
    assert sys.modules[module_name] is module
    assert sys.modules[qualified] is module

    package = sys.modules.get("menace_sandbox")
    assert package is not None
    pkg_path = list(getattr(package, "__path__", []))
    assert str(repo_root) in pkg_path
