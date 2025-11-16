import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.error_bot as eb  # noqa: E402
import menace.database_steward_bot as dsb  # noqa: E402
import menace.conversation_manager_bot as cmb  # noqa: E402
import menace.chatgpt_idea_bot as cib  # noqa: E402
import menace.data_bot as db  # noqa: E402


def test_bot_safe_mode(tmp_path):
    err_db = eb.ErrorDB(tmp_path / "e.db")

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, query, **_):
            return ""

    builder = DummyBuilder()
    conv = cmb.ConversationManagerBot(cib.ChatGPTClient("key", context_builder=builder))
    bot = eb.ErrorBot(
        err_db,
        conversation_bot=conv,
        metrics_db=db.MetricsDB(tmp_path / "m.db"),
        context_builder=builder,
    )
    steward = dsb.DatabaseStewardBot(
        sql_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        error_bot=bot,
        conversation_bot=conv,
    )
    bot.record_runtime_error("boom", bot_ids=["DatabaseStewardBot"])
    with pytest.raises(RuntimeError):
        steward.deduplicate()
    bot.clear_module_flag("DatabaseStewardBot")
    assert not err_db.is_safe_mode("DatabaseStewardBot")
    steward.deduplicate()
