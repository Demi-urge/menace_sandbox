import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import coding_bot_interface
import code_database
import vector_metrics_db
import quick_fix_engine
import prompt_engine
from code_database import PatchHistoryDB
from prompt_memory_trainer import PromptMemoryTrainer


def test_prepare_pipeline_with_model_automation_pipeline():
    class ModelAutomationPipeline:
        def __init__(
            self,
            *,
            context_builder=None,
            bot_registry=None,
            data_bot=None,
            manager=None,
        ) -> None:
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.manager = manager

    pipeline, promote = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=ModelAutomationPipeline,
        context_builder=object(),
        bot_registry=type("Registry", (), {"set_bootstrap_mode": lambda *_: None})(),
        data_bot=object(),
    )

    promote(object())

    assert isinstance(pipeline, ModelAutomationPipeline)


def test_prepare_pipeline_fast_path_skips_vector_lock(monkeypatch, tmp_path):
    locked_db = tmp_path / "vector_metrics.db"
    locked_db.touch()
    lock_conn = sqlite3.connect(locked_db)
    lock_conn.execute("BEGIN EXCLUSIVE")

    monkeypatch.setattr(
        vector_metrics_db, "default_vector_metrics_path", lambda ensure_exists=True: locked_db
    )

    class FastPipeline:
        def __init__(
            self,
            *,
            context_builder=None,
            bot_registry=None,
            data_bot=None,
            manager=None,
            bootstrap_fast=False,
            **_: object,
        ) -> None:
            self.manager = manager
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.trainer = PromptMemoryTrainer(
                patch_db=PatchHistoryDB(tmp_path / "patch_history.db", bootstrap=bootstrap_fast),
                bootstrap_fast=bootstrap_fast,
            )

    start = time.perf_counter()
    pipeline, promote = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=FastPipeline,
        context_builder=object(),
        bot_registry=SimpleNamespace(set_bootstrap_mode=lambda *_: None),
        data_bot=object(),
        bootstrap_safe=True,
        bootstrap_fast=True,
    )
    elapsed = time.perf_counter() - start

    try:
        promote(object())
    finally:
        lock_conn.rollback()
        lock_conn.close()

    assert pipeline is not None
    assert elapsed < 5


def test_prepare_pipeline_survives_locked_patch_db(monkeypatch, tmp_path):
    locked_db = tmp_path / "patch_history.db"
    lock_conn = sqlite3.connect(locked_db)
    lock_conn.execute("BEGIN EXCLUSIVE")

    class FastPipeline:
        def __init__(
            self,
            *,
            context_builder=None,
            bot_registry=None,
            data_bot=None,
            manager=None,
            bootstrap_fast=False,
            **_: object,
        ) -> None:
            self.manager = manager
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.trainer = PromptMemoryTrainer(
                patch_db=PatchHistoryDB(locked_db, bootstrap=bootstrap_fast),
                bootstrap_fast=bootstrap_fast,
            )

    start = time.perf_counter()
    pipeline, promote = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=FastPipeline,
        context_builder=object(),
        bot_registry=SimpleNamespace(set_bootstrap_mode=lambda *_: None),
        data_bot=object(),
        bootstrap_safe=True,
    )
    elapsed = time.perf_counter() - start

    try:
        promote(object())
    finally:
        lock_conn.rollback()
        lock_conn.close()

    assert pipeline is not None
    assert elapsed < 5


def test_shared_vector_service_short_circuits_schema_discovery(monkeypatch, tmp_path):
    locked_db = tmp_path / "vector_metrics.db"
    locked_db.touch()
    lock_conn = sqlite3.connect(locked_db)
    lock_conn.execute("BEGIN EXCLUSIVE")

    monkeypatch.setenv("VECTOR_SERVICE_LAZY_BOOTSTRAP", "1")
    monkeypatch.setenv("VECTOR_SERVICE_SKIP_DISCOVERY", "1")
    monkeypatch.setattr(
        vector_metrics_db, "default_vector_metrics_path", lambda ensure_exists=True: locked_db
    )

    from vector_service import registry as vec_registry
    from vector_service.vectorizer import SharedVectorService

    monkeypatch.setattr(
        vec_registry,
        "_VECTOR_REGISTRY",
        {
            "patch": (
                "vector_service.patch_vectorizer",
                "PatchVectorizer",
                None,
                None,
            )
        },
    )

    class Pipeline:
        def __init__(
            self,
            *,
            context_builder=None,
            bot_registry=None,
            data_bot=None,
            manager=None,
            bootstrap_fast=False,
            **_: object,
        ) -> None:
            self.manager = manager
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            embedder = SimpleNamespace(encode=lambda items: [[0.0] * 2 for _ in items])
            store = SimpleNamespace(add=lambda *_a, **_k: None)
            self.vector_service = SharedVectorService(
                text_embedder=embedder,
                vector_store=store,
                bootstrap_fast=bootstrap_fast,
            )

    start = time.perf_counter()
    pipeline, promote = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=Pipeline,
        context_builder=object(),
        bot_registry=SimpleNamespace(set_bootstrap_mode=lambda *_: None),
        data_bot=object(),
        bootstrap_safe=True,
        bootstrap_fast=True,
    )
    elapsed = time.perf_counter() - start

    try:
        promote(object())
    finally:
        lock_conn.rollback()
        lock_conn.close()

    assert pipeline is not None
    assert elapsed < 5


def test_bootstrap_fast_skips_patch_schema(monkeypatch, tmp_path):
    statements: list[str] = []

    class StubConn:
        def execute(self, sql: str, *args, **kwargs):  # noqa: ARG002
            statements.append(sql.strip())
            normalised = sql.lstrip().upper()
            if normalised.startswith("PRAGMA") or normalised.startswith("CREATE"):
                raise AssertionError(f"Schema query executed: {sql}")
            return SimpleNamespace(fetchall=lambda: [])

        def commit(self) -> None:
            pass

        def rollback(self) -> None:
            pass

    stub_conn = StubConn()

    class StubRouter:
        def get_connection(self, _name: str):
            return stub_conn

    monkeypatch.setattr(code_database, "_PATCH_HISTORY_BOOTSTRAP_ROUTER", None)
    monkeypatch.setattr(code_database, "init_db_router", lambda *args, **kwargs: StubRouter())

    class FastPipeline:
        def __init__(
            self,
            *,
            context_builder=None,
            bot_registry=None,
            data_bot=None,
            manager=None,
            bootstrap_fast=False,
            **_: object,
        ) -> None:
            self.manager = manager
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.trainer = PromptMemoryTrainer(
                patch_db=PatchHistoryDB(tmp_path / "patch_history.db", bootstrap_fast=bootstrap_fast),
                bootstrap_fast=bootstrap_fast,
            )

    pipeline, promote = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=FastPipeline,
        context_builder=object(),
        bot_registry=SimpleNamespace(set_bootstrap_mode=lambda *_: None),
        data_bot=object(),
        bootstrap_safe=True,
        bootstrap_fast=True,
    )

    promote(object())

    assert pipeline is not None
    assert not statements


def test_quick_fix_bootstrap_skips_slow_metrics(monkeypatch):
    slow_calls: list[tuple[bool, bool]] = []

    class SlowContextBuilder:
        def refresh_db_weights(
            self, *, bootstrap: bool = False, bootstrap_fast: bool = False, **_: object
        ) -> dict[str, float]:
            slow_calls.append((bootstrap, bootstrap_fast))
            if not (bootstrap or bootstrap_fast):
                raise TimeoutError("refresh_db_weights should not block during bootstrap")
            return {"code": 1.0}

    class StubRegistry:
        bootstrap = True
        bootstrap_fast = True

        def register_bot(self, *_: object, **__: object) -> None:
            return None

        def set_bootstrap_mode(self, *_: object, **__: object) -> None:
            return None

    class StubErrorDB:
        pass

    manager = SimpleNamespace(
        bootstrap=True,
        bootstrap_fast=True,
        bootstrap_runtime=True,
        bot_registry=StubRegistry(),
        data_bot=object(),
    )

    builder = SlowContextBuilder()

    monkeypatch.setattr(
        quick_fix_engine, "_get_knowledge_graph_cls", lambda: type("KG", (), {})
    )
    monkeypatch.setattr(
        quick_fix_engine, "_get_patch_logger_cls", lambda: lambda **_: SimpleNamespace()
    )

    class Pipeline:
        def __init__(
            self,
            *,
            context_builder=None,
            bot_registry=None,
            data_bot=None,
            manager=None,
            bootstrap_fast=False,
            **_: object,
        ) -> None:
            self.manager = manager
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.quick_fix = quick_fix_engine.QuickFixEngine(
                StubErrorDB(),
                manager,
                context_builder=context_builder,
                patch_logger=SimpleNamespace(),
                helper_fn=lambda *_a, **_k: "",
            )

    start = time.perf_counter()
    pipeline, promote = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=Pipeline,
        context_builder=builder,
        bot_registry=manager.bot_registry,
        data_bot=manager.data_bot,
        manager_override=manager,
        bootstrap_safe=True,
        bootstrap_fast=True,
    )
    elapsed = time.perf_counter() - start

    promote(manager)

    assert pipeline is not None
    assert slow_calls and slow_calls[-1][0] and slow_calls[-1][1]
    assert elapsed < 1


def test_prepare_pipeline_skips_prompt_weights_resolution(monkeypatch, tmp_path):
    slow_weights = tmp_path / "slow" / "prompt_style_weights.json"

    real_resolve = prompt_engine.resolve_path

    def guarded_resolve(name, *args, **kwargs):
        if Path(name) == slow_weights:
            raise AssertionError("weights_path resolution should be skipped during bootstrap")
        return real_resolve(name, *args, **kwargs)

    monkeypatch.setattr(prompt_engine, "resolve_path", guarded_resolve)

    class PromptPipeline:
        def __init__(
            self,
            *,
            context_builder=None,
            bot_registry=None,
            data_bot=None,
            manager=None,
            bootstrap_fast=False,
            **_: object,
        ) -> None:
            self.manager = manager
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.prompt_engine = prompt_engine.PromptEngine(
                context_builder=context_builder,
                retriever=None,
                weights_path=slow_weights,
                bootstrap_fast=bootstrap_fast,
            )

    pipeline, promote = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=PromptPipeline,
        context_builder=object(),
        bot_registry=SimpleNamespace(set_bootstrap_mode=lambda *_: None),
        data_bot=object(),
        bootstrap_safe=True,
        bootstrap_fast=True,
    )

    promote(object())

    assert pipeline is not None


def test_prepare_pipeline_bootstrap_fast_paths_prompt_engine(monkeypatch, tmp_path):
    slow_weights = tmp_path / "slow" / "prompt_style_weights.json"
    slow_template = tmp_path / "slow" / "prompt_templates.json"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    real_resolve = prompt_engine.resolve_path
    resolved: list[Path] = []

    def tracking_resolve(path, *args, **kwargs):  # noqa: ANN001
        resolved.append(Path(path))
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(prompt_engine, "resolve_path", tracking_resolve)

    class PromptPipeline:
        def __init__(
            self,
            *,
            context_builder=None,
            bot_registry=None,
            data_bot=None,
            manager=None,
            bootstrap_fast=False,
            **_: object,
        ) -> None:
            self.manager = manager
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.prompt_engine = prompt_engine.PromptEngine(
                context_builder=SimpleNamespace(roi_tracker=None),
                retriever=None,
                weights_path=slow_weights,
                template_path=slow_template,
                chunk_summary_cache_dir=cache_dir,
                bootstrap_fast=bootstrap_fast,
            )

    start = time.perf_counter()
    pipeline, promote = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=PromptPipeline,
        context_builder=object(),
        bot_registry=SimpleNamespace(set_bootstrap_mode=lambda *_: None),
        data_bot=object(),
        bootstrap_safe=True,
        bootstrap_fast=True,
    )
    elapsed = time.perf_counter() - start

    promote(object())

    assert pipeline is not None
    assert slow_weights not in resolved
    assert slow_template not in resolved
    assert elapsed < 5
    assert pipeline.prompt_engine.bootstrap_fast is True
    assert pipeline.prompt_engine.weights_path == slow_weights
