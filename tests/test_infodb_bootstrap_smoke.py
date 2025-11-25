import sqlite3
import time

from menace_sandbox import coding_bot_interface
from menace_sandbox.research_aggregator_bot import InfoDB, ResearchAggregatorBot


class _StubBuilder:
    def refresh_db_weights(self) -> None:
        return None


class _StubRegistry:
    pass


class _StubDataBot:
    pass


def test_prepare_pipeline_handles_locked_infodb(tmp_path) -> None:
    db_path = tmp_path / "locked.db"
    lock_conn = sqlite3.connect(db_path)
    lock_conn.execute("BEGIN EXCLUSIVE")

    class _LockedPipeline:
        def __init__(self, *, context_builder, bot_registry, data_bot, manager=None, **_kwargs):
            self.manager = manager
            self.context_builder = context_builder
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.info_db = InfoDB(
                path=db_path,
                vector_index_path=tmp_path / "info.idx",
                batch_migrations=True,
                bootstrap_mode=bool(getattr(manager, "bootstrap_mode", False)),
                migration_timeout=0.0,
                non_blocking_migrations=True,
            )
            self.aggregator = ResearchAggregatorBot(
                ["topic"],
                info_db=self.info_db,
                context_builder=context_builder,
                manager=manager,
                bootstrap=True,
                defer_migrations_until_ready=True,
            )

    start = time.perf_counter()
    pipeline, _promote = coding_bot_interface.prepare_pipeline_for_bootstrap(
        pipeline_cls=_LockedPipeline,
        context_builder=_StubBuilder(),
        bot_registry=_StubRegistry(),
        data_bot=_StubDataBot(),
        bootstrap_safe=True,
    )
    elapsed = time.perf_counter() - start
    lock_conn.rollback()
    lock_conn.close()

    assert pipeline is not None
    assert elapsed < 1.0
