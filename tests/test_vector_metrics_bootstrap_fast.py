import sqlite3
import time
from types import SimpleNamespace

import coding_bot_interface
import code_database
import vector_metrics_db
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
