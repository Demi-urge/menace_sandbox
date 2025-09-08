import json
import sys
import types


def test_anomaly_threshold_triggers_engine(monkeypatch, tmp_path):
    """Repeated anomalies log to memory and trigger engine update."""

    # Stub dynamic path resolution to a temporary directory
    dynamic_stub = types.SimpleNamespace(
        resolve_path=lambda p: tmp_path / p,
        resolve_dir=lambda p: tmp_path / p,
        get_project_root=lambda: tmp_path,
        get_project_roots=lambda: [tmp_path],
    )
    monkeypatch.setitem(sys.modules, "dynamic_path_router", dynamic_stub)

    # Stub heavy discrepancy_db dependencies before imports
    discrepancy_stub = types.ModuleType("discrepancy_db")

    class _DummyDB:
        def __init__(self, *a, **k):
            pass

    discrepancy_stub.DiscrepancyDB = _DummyDB
    discrepancy_stub.DiscrepancyRecord = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "discrepancy_db", discrepancy_stub)

    fls_stub = types.ModuleType("failure_learning_system")
    fls_stub.DiscrepancyDB = _DummyDB  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "failure_learning_system", fls_stub)

    # Provide minimal event bus implementation for menace_sanity_layer fallback
    (tmp_path / "unified_event_bus.py").write_text(
        "class UnifiedEventBus:\n"
        "    def __init__(self):\n"
        "        self.handlers = []\n"
        "    def subscribe(self, topic, cb):\n"
        "        self.handlers.append((topic, cb))\n"
        "    def publish(self, topic, event):\n"
        "        for t, cb in list(self.handlers):\n"
        "            if t == topic:\n"
        "                cb(topic, event)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    from dynamic_path_router import resolve_path
    from db_router import init_db_router

    # Initialise router to avoid filesystem lookups during imports
    init_db_router(
        "ba",
        str(resolve_path("local.db")),
        str(resolve_path("shared.db")),
    )

    import menace_sanity_layer as msl
    import stripe_watchdog as sw

    # Avoid external side effects
    monkeypatch.setattr(sw, "_refresh_instruction_cache", lambda: None)
    monkeypatch.setattr(sw.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(msl.audit_logger, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(sw.ANOMALY_TRAIL, "record", lambda *a, **k: None)
    monkeypatch.setattr(sw, "record_event", lambda *a, **k: None)
    monkeypatch.setattr(sw, "record_billing_anomaly", lambda *a, **k: None)

    # Stub GPT memory manager to capture entries
    class DummyMemory:
        def __init__(self) -> None:
            self.entries: list[tuple[str, dict, list[str]]] = []

        def log_interaction(self, instruction, content, *, tags=None):
            self.entries.append((instruction, json.loads(content), tags or []))

    memory = DummyMemory()
    monkeypatch.setattr(msl, "GPT_MEMORY_MANAGER", memory)

    # Stub MenaceMemoryManager used for threshold tracking
    mm_storage: dict[str, dict] = {}

    class DummyMM:
        def query(self, key, limit):
            if key in mm_storage:
                return [types.SimpleNamespace(data=json.dumps(mm_storage[key]))]
            return []

        def store(self, key, data, tags=""):
            mm_storage[key] = data

    monkeypatch.setattr(msl, "_get_memory_manager", lambda: DummyMM())
    monkeypatch.setattr(msl, "_DISCREPANCY_DB", None)

    # Capture generation parameter updates
    class DummyEngine:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def update_generation_params(self, meta):
            self.calls.append(meta)

    engine = DummyEngine()

    record = {"type": "missing_charge", "id": "ch_1"}
    threshold = msl.PAYMENT_ANOMALY_THRESHOLD
    monkeypatch.setattr(sw, "record_billing_event", lambda *a, **k: None)
    monkeypatch.setattr(sw, "load_api_key", lambda: None)
    builder = types.SimpleNamespace(build=lambda *a, **k: "")

    for _ in range(threshold + 1):
        sw._emit_anomaly(
            record, False, False, self_coding_engine=engine, context_builder=builder
        )

    # Memory receives an entry for each anomaly
    assert len(memory.entries) == threshold + 1

    # Engine receives a single hint once the threshold is crossed
    expected_hint = {**msl.ANOMALY_HINTS["missing_charge"], "event_type": "missing_charge"}
    assert engine.calls == [expected_hint]
