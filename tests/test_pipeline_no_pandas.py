import builtins
import importlib
import logging
import sys
import types

import dependency_health


def _load_pipeline_base(monkeypatch):
    stub_vector_service = types.ModuleType("vector_service")
    stub_vector_service.__path__ = []
    monkeypatch.setitem(sys.modules, "vector_service", stub_vector_service)

    context_builder_mod = types.ModuleType("vector_service.context_builder")

    class ContextBuilder:
        def refresh_db_weights(self) -> None:
            return None

    context_builder_mod.ContextBuilder = ContextBuilder
    monkeypatch.setitem(
        sys.modules, "vector_service.context_builder", context_builder_mod
    )

    st_module = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            return None

    st_module.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_module)

    transformers_module = types.ModuleType("transformers")

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    transformers_module.AutoModel = _AutoModel
    transformers_module.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    torch_module = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    return importlib.import_module("menace.shared.pipeline_base")


def test_pipeline_handles_missing_pandas(monkeypatch):
    pipeline_base = _load_pipeline_base(monkeypatch)
    from menace.synthesis_models import SynthesisTask

    monkeypatch.setattr(pipeline_base, "pd", None, raising=False)

    pipeline = pipeline_base.ModelAutomationPipeline.__new__(
        pipeline_base.ModelAutomationPipeline
    )
    pipeline.logger = logging.getLogger("pipeline-test")
    pipeline._create_synthesis_task = lambda **kwargs: SynthesisTask(**kwargs)

    captured_rows = {}

    class DummySynthesisBot:
        def create_tasks(self, table):
            rows = list(table.iterrows())
            captured_rows["rows"] = rows
            return [
                SynthesisTask(
                    description=row.get("content", ""),
                    urgency=1,
                    complexity=1,
                    category="analysis",
                )
                for _, row in rows
            ]

    pipeline.synthesis_bot = DummySynthesisBot()
    validator = types.SimpleNamespace(validate=lambda task: True)
    monkeypatch.setattr(pipeline, "_ensure_validator", lambda: validator)

    base_tasks = [
        SynthesisTask(description="alpha", urgency=1, complexity=1, category="analysis")
    ]

    table = pipeline._build_task_table(base_tasks)
    assert hasattr(table, "iterrows")

    refined = pipeline.synthesis_bot.create_tasks(table)
    validated = pipeline._validate_tasks(refined)
    assert len(validated) == 1
    assert captured_rows["rows"][0][1]["content"] == "alpha"

    dict_validated = pipeline._validate_tasks(
        [
            {
                "description": "beta",
                "urgency": 2,
                "complexity": "3",
                "category": "analysis",
            }
        ]
    )
    assert len(dict_validated) == 1
    assert getattr(dict_validated[0], "description", None) == "beta"


def test_pipeline_reports_pandas_import_errors(monkeypatch, caplog):
    new_registry = dependency_health.DependencyHealthRegistry()
    monkeypatch.setattr(
        dependency_health, "dependency_registry", new_registry, raising=False
    )
    monkeypatch.delitem(sys.modules, "menace.shared.pipeline_base", raising=False)
    monkeypatch.delitem(
        sys.modules, "menace_sandbox.shared.pipeline_base", raising=False
    )
    monkeypatch.delitem(sys.modules, "pandas", raising=False)

    error_message = "simulated pandas import failure"

    original_import = builtins.__import__

    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pandas":
            sys.modules[name] = types.ModuleType("pandas_poison")
            raise RuntimeError(error_message)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    caplog.set_level(logging.WARNING)
    pipeline_base = _load_pipeline_base(monkeypatch)

    assert pipeline_base.pd is None

    summary = dependency_health.dependency_registry.summary()
    pandas_entries = [
        entry for entry in summary.get("missing", []) if entry.get("name") == "pandas"
    ]
    assert pandas_entries
    assert error_message in pandas_entries[0].get("reason", "")
    assert error_message in caplog.text
    monkeypatch.delitem(sys.modules, "pandas", raising=False)
