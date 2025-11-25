import sqlite3
import time
from types import SimpleNamespace

import coding_bot_interface
import vector_metrics_db
from code_database import PatchHistoryDB
from prompt_memory_trainer import PromptMemoryTrainer


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
