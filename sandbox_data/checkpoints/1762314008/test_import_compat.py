from __future__ import annotations

import importlib
import sys
from pathlib import Path
from textwrap import dedent
from types import ModuleType, SimpleNamespace

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
    "self_coding_engine",
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


@pytest.fixture
def stub_external_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    stub_root = tmp_path / "import_stubs"
    stub_root.mkdir()

    vector_service = stub_root / "vector_service"
    roi_tags = vector_service / "roi_tags.py"
    vector_service.mkdir()
    (vector_service / "__init__.py").write_text(
        dedent(
            """
            class VectorServiceError(Exception):
                'Minimal stub used in tests.'

            class ContextBuilder:
                def __init__(self, *args, **kwargs):
                    self.roi_tracker = None

                def refresh_db_weights(self) -> None:
                    return None

            class CognitionLayer:
                def __init__(self) -> None:
                    self.context_builder = ContextBuilder()

            class PatchLogger:
                def track_contributors(self, *args, **kwargs) -> None:
                    return None

            class SharedVectorService:
                def __init__(self, *args, **kwargs) -> None:
                    self.args = args
                    self.kwargs = kwargs

                def embed(self, *args, **kwargs) -> list[float]:
                    return []
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    roi_tags.write_text(
        dedent(
            """
            from enum import Enum


            class RoiTag(Enum):
                SUCCESS = "success"
                HIGH_ROI = "high"
                LOW_ROI = "low"

                @classmethod
                def validate(cls, value):
                    if isinstance(value, cls):
                        return value
                    for member in cls:
                        if member.value == value:
                            return member
                    return cls.SUCCESS
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    sklearn = stub_root / "sklearn"
    sklearn.mkdir()
    (sklearn / "__init__.py").write_text("", encoding="utf-8")
    (sklearn / "cluster.py").write_text(
        dedent(
            """
            class _Result(list):
                def tolist(self):
                    return list(self)


            class KMeans:
                def __init__(self, n_clusters, **kwargs):
                    self.n_clusters = max(1, int(n_clusters))

                def fit_predict(self, matrix):
                    size = len(matrix)
                    return _Result((index % self.n_clusters) for index in range(size))
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    feature_extraction = sklearn / "feature_extraction"
    feature_extraction.mkdir()
    (feature_extraction / "__init__.py").write_text("", encoding="utf-8")
    (feature_extraction / "text.py").write_text(
        dedent(
            """
            class TfidfVectorizer:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def fit_transform(self, documents):
                    return list(documents)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    prompt_engine_stub = ModuleType("menace_sandbox.prompt_engine")

    class _PromptEngine:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.last_metadata = {}

        def render(self, *args, **kwargs):
            return SimpleNamespace(text="", metadata={})

    def _diff_within_target_region(*_args, **_kwargs):
        return True

    prompt_engine_stub.PromptEngine = _PromptEngine
    prompt_engine_stub._ENCODER = None
    prompt_engine_stub.diff_within_target_region = _diff_within_target_region

    sys.modules["menace_sandbox.prompt_engine"] = prompt_engine_stub
    sys.modules["prompt_engine"] = prompt_engine_stub

    test_harness_stub = ModuleType("menace_sandbox.sandbox_runner.test_harness")

    class _TestHarnessResult(SimpleNamespace):
        success = True
        stdout = ""

    def _run_tests(*_args, **_kwargs):
        return _TestHarnessResult()

    test_harness_stub.run_tests = _run_tests
    test_harness_stub.TestHarnessResult = _TestHarnessResult

    sys.modules["menace_sandbox.sandbox_runner.test_harness"] = test_harness_stub
    sys.modules["sandbox_runner.test_harness"] = test_harness_stub

    error_bot_stub = ModuleType("menace_sandbox.error_bot")

    class _ErrorDB:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.rows = []

    error_bot_stub.ErrorDB = _ErrorDB

    sys.modules["menace_sandbox.error_bot"] = error_bot_stub
    sys.modules["error_bot"] = error_bot_stub

    scm_stub = ModuleType("menace_sandbox.self_coding_manager")

    class _SelfCodingManager(SimpleNamespace):
        def __init__(self, *args, **kwargs):
            layer = SimpleNamespace(context_builder=SimpleNamespace())
            super().__init__(
                engine=SimpleNamespace(cognition_layer=layer),
                data_bot=kwargs.get("data_bot"),
                bot_registry=kwargs.get("bot_registry"),
                quick_fix=kwargs.get("quick_fix"),
            )

    def _manager_helper(*_args, **_kwargs):
        return ""

    scm_stub.SelfCodingManager = _SelfCodingManager
    scm_stub._manager_generate_helper_with_builder = _manager_helper

    sys.modules["menace_sandbox.self_coding_manager"] = scm_stub
    sys.modules["self_coding_manager"] = scm_stub

    prompt_strategies_stub = ModuleType("menace_sandbox.self_improvement.prompt_strategies")

    class _PromptStrategy(SimpleNamespace):
        pass

    def _render_prompt(*_args, **_kwargs):
        return ""

    prompt_strategies_stub.PromptStrategy = _PromptStrategy
    prompt_strategies_stub.render_prompt = _render_prompt

    sys.modules["menace_sandbox.self_improvement.prompt_strategies"] = prompt_strategies_stub
    sys.modules["self_improvement.prompt_strategies"] = prompt_strategies_stub

    monkeypatch.syspath_prepend(str(stub_root))
    return stub_root


@pytest.mark.parametrize("module_name", MODULES)
def test_flat_import_aliases(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    stub_external_dependencies: Path,
) -> None:
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


@pytest.mark.parametrize("module_name", MODULES)
def test_package_import_aliases(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    stub_external_dependencies: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))

    qualified = f"menace_sandbox.{module_name}"
    for key in (module_name, qualified, "menace_sandbox", "menace_sandbox.import_compat", "import_compat"):
        sys.modules.pop(key, None)

    try:
        module = importlib.import_module(qualified)
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

    assert sys.modules[qualified] is module
    assert sys.modules[module_name] is module
