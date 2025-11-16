import types
import pytest
import menace.error_bot as eb
import menace.self_coding_manager as scm
import menace.data_bot as db


class DummyBuilder:
    def refresh_db_weights(self):
        pass


class DummyEngine:
    pass


class DummyPipeline:
    pass


class DummyDataBot:
    def roi(self, _bot: str) -> float:
        return 1.0

    def average_errors(self, _bot: str) -> float:
        return 0.0

    def average_test_failures(self, _bot: str) -> float:
        return 0.0

    def get_thresholds(self, _bot: str):
        return types.SimpleNamespace(
            roi_drop=-1.0, error_threshold=1.0, test_failure_threshold=1.0
        )


class DummyRegistry:
    def register_bot(self, name: str) -> None:
        pass


def test_error_bot_requires_manager(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    with pytest.raises(TypeError):
        eb.ErrorBot(eb.ErrorDB(tmp_path / "e.db"), mdb, context_builder=DummyBuilder())
    manager = scm.SelfCodingManager(
        DummyEngine(),
        DummyPipeline(),
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
    )
    bot = eb.ErrorBot(
        eb.ErrorDB(tmp_path / "e.db"),
        mdb,
        context_builder=DummyBuilder(),
        selfcoding_manager=manager,
    )
    assert bot.manager is manager
