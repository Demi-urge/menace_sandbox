import sys
import types
from unittest.mock import patch

import db_router

sys.modules.setdefault(
    "menace.failure_learning_system",
    types.SimpleNamespace(
        DiscrepancyDB=object,
        FailureRecord=object,
        FailureLearningSystem=object,
    ),
)
sys.modules.setdefault("menace.error_bot", types.SimpleNamespace(ErrorDB=object))
sys.modules.setdefault("menace.databases", types.SimpleNamespace(MenaceDB=object))
sys.modules.setdefault(
    "menace.unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object)
)
sys.modules.setdefault(
    "menace.menace_memory_manager",
    types.SimpleNamespace(MenaceMemoryManager=object, MemoryEntry=object),
)
sys.modules.setdefault(
    "menace.performance_assessment_bot",
    types.SimpleNamespace(SimpleRL=object),
)
sys.modules.setdefault(
    "menace.contextual_rl", types.SimpleNamespace(ContextualRL=object)
)
sys.modules.setdefault(
    "menace.evolution_history_db",
    types.SimpleNamespace(EvolutionHistoryDB=object, EvolutionEvent=object),
)
sys.modules.setdefault("menace.data_bot", types.SimpleNamespace(MetricsDB=object))
sys.modules.setdefault(
    "menace.cross_query", types.SimpleNamespace(workflow_roi_stats=lambda *a, **k: {})
)
sys.modules.setdefault(
    "menace.retry_utils", types.SimpleNamespace(retry=lambda f: f)
)

import menace.resource_allocation_optimizer as rao  # noqa: E402


def test_roidb_routes_through_router(tmp_path):
    router = db_router.DBRouter("roi", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    with patch.object(router, "get_connection", wraps=router.get_connection) as gc:
        db = rao.ROIDB(path=tmp_path / "roi.db", router=router)
        db.add(rao.KPIRecord("b", 1.0, 1.0, 1.0, 1.0))
        db.add_action_roi("act", 1.0, 1.0, 1.0, 1.0)
        db.add_weight("b", 0.5)
        calls = {c.args[0] for c in gc.call_args_list}
    assert {"roi", "action_roi", "allocation_weights"} <= calls
    conn = router.get_connection("roi")
    path = conn.execute("PRAGMA database_list").fetchone()[2]
    assert path == str(tmp_path / "local.db")
    assert {"roi", "action_roi", "allocation_weights"} <= db_router.LOCAL_TABLES
    router.close()
