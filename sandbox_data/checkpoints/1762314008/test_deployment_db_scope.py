import sys
from types import SimpleNamespace

# Stub heavy dependencies before importing DeploymentDB
sys.modules.setdefault("menace.data_bot", SimpleNamespace())
sys.modules.setdefault("menace.capital_management_bot", SimpleNamespace(CapitalManagementBot=object))
sys.modules.setdefault("menace.database_management_bot", SimpleNamespace(DatabaseManagementBot=object))
sys.modules.setdefault("menace.chatgpt_idea_bot", SimpleNamespace(ChatGPTClient=object))
sys.modules.setdefault(
    "menace.chatgpt_enhancement_bot",
    SimpleNamespace(EnhancementDB=object, ChatGPTEnhancementBot=object, Enhancement=object),
)
sys.modules.setdefault("menace.research_aggregator_bot", SimpleNamespace(InfoDB=object))
sys.modules.setdefault("menace.error_bot", SimpleNamespace(ErrorDB=object))
sys.modules.setdefault("menace.deployment_governance", SimpleNamespace(evaluate_scorecard=lambda *a, **k: None))
sys.modules.setdefault("menace.governance", SimpleNamespace(evaluate_rules=lambda *a, **k: None))

import menace.deployment_bot as db


def test_errors_for_scope(tmp_path):
    ddb = db.DeploymentDB(tmp_path / "dep.db")
    menace_id = ddb.router.menace_id
    other_id = "other"
    conn = ddb.router.get_connection("errors")
    cur = conn.execute(
        "INSERT INTO errors(source_menace_id, deploy_id, message, ts) VALUES (?,?,?,?)",
        (menace_id, 1, "local", "2020"),
    )
    local_id = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO errors(source_menace_id, deploy_id, message, ts) VALUES (?,?,?,?)",
        (other_id, 1, "remote", "2020"),
    )
    remote_id = cur.lastrowid
    conn.commit()

    assert ddb.errors_for(1, scope="local") == [local_id]
    assert ddb.errors_for(1, scope="global") == [remote_id]
    assert set(ddb.errors_for(1, scope="all")) == {local_id, remote_id}
