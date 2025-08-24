import sys
from types import SimpleNamespace

# Stub modules with heavy dependencies before import
sys.modules.setdefault("menace.chatgpt_idea_bot", SimpleNamespace(ChatGPTClient=object))
sys.modules.setdefault("menace.database_management_bot", SimpleNamespace(DatabaseManagementBot=object))
sys.modules.setdefault("menace.capital_management_bot", SimpleNamespace(CapitalManagementBot=object))
sys.modules.setdefault("menace.data_bot", SimpleNamespace())

# Ensure we import the real enhancement DB after stubbing dependencies
sys.modules.pop("menace.chatgpt_enhancement_bot", None)
import menace.chatgpt_enhancement_bot as ceb


def test_scope(tmp_path):
    db = ceb.EnhancementDB(tmp_path / "e.db")
    db.conn.execute(
        "INSERT INTO enhancements(idea, rationale, source_menace_id) VALUES (?,?,?)",
        ("a", "r", db.router.menace_id),
    )
    db.conn.execute(
        "INSERT INTO enhancements(idea, rationale, source_menace_id) VALUES (?,?,?)",
        ("b", "r", "other"),
    )
    db.conn.commit()
    assert {e.idea for e in db.fetch(scope="local")} == {"a"}
    assert {e.idea for e in db.fetch(scope="global")} == {"b"}
    assert {e.idea for e in db.fetch(scope="all")} == {"a", "b"}
