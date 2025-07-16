import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import pytest
import menace.error_bot as eb
import menace.database_steward_bot as dsb
import menace.conversation_manager_bot as cmb
import menace.chatgpt_idea_bot as cib
import menace.data_bot as db


def test_bot_safe_mode(tmp_path):
    err_db = eb.ErrorDB(tmp_path / "e.db")
    conv = cmb.ConversationManagerBot(cib.ChatGPTClient("key"))
    bot = eb.ErrorBot(err_db, conversation_bot=conv, metrics_db=db.MetricsDB(tmp_path / "m.db"))
    steward = dsb.DatabaseStewardBot(sql_url=f"sqlite:///{tmp_path / 'db.sqlite'}", error_bot=bot, conversation_bot=conv)
    bot.record_runtime_error("boom", bot_ids=["DatabaseStewardBot"])
    with pytest.raises(RuntimeError):
        steward.deduplicate()
    bot.clear_module_flag("DatabaseStewardBot")
    assert not err_db.is_safe_mode("DatabaseStewardBot")
    steward.deduplicate()
