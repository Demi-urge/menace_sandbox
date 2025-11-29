+import types
+
+import pytest
+
+import single_agent_roi_runner as runner
+
+
+def test_roi_runner_reuses_broker_pipeline(monkeypatch):
+    broker_pipeline = object()
+    broker_manager = object()
+    broker = types.SimpleNamespace(resolve=lambda: (broker_pipeline, broker_manager))
+
+    monkeypatch.setattr(runner, "_bootstrap_dependency_broker", lambda: broker)
+    monkeypatch.setattr(runner, "_current_bootstrap_context", lambda: None)
+    monkeypatch.setattr(runner, "_looks_like_pipeline_candidate", lambda value: bool(value))
+    monkeypatch.setattr(runner, "_BOOTSTRAP_STATE", types.SimpleNamespace(helper_promotion_callbacks=[], depth=1))
+
+    monkeypatch.setattr(runner, "BotRegistry", lambda *a, **k: object())
+    monkeypatch.setattr(runner, "DataBot", lambda *a, **k: object())
+    monkeypatch.setattr(runner, "create_context_builder", lambda *a, **k: object())
+    monkeypatch.setattr(runner, "SelfCodingEngine", lambda *a, **k: object())
+    monkeypatch.setattr(runner, "CodeDB", lambda *a, **k: object())
+    monkeypatch.setattr(runner, "MenaceMemoryManager", lambda *a, **k: object())
+    monkeypatch.setattr(
+        runner,
+        "get_thresholds",
+        lambda *_a, **_k: types.SimpleNamespace(
+            roi_drop=0.0, error_increase=0.0, test_failure_increase=0.0
+        ),
+    )
+    monkeypatch.setattr(runner, "ThresholdService", lambda *a, **k: object())
+
+    internalize_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
+
+    def _record_internalize(*args, **kwargs):
+        internalize_calls.append((args, kwargs))
+        return broker_manager
+
+    monkeypatch.setattr(runner, "internalize_coding_bot", _record_internalize)
+
+    bootstrap = runner.build_manager("ROI", dependency_broker=broker)
+
+    assert bootstrap.pipeline is broker_pipeline
+    assert bootstrap.manager is broker_manager
+    assert not internalize_calls
+
+
+def test_roi_runner_blocks_bootstrap_without_pipeline(monkeypatch):
+    monkeypatch.setattr(runner, "_bootstrap_dependency_broker", lambda: types.SimpleNamespace(resolve=lambda: (None, None)))
+    monkeypatch.setattr(runner, "_current_bootstrap_context", lambda: None)
+    monkeypatch.setattr(runner, "_looks_like_pipeline_candidate", lambda value: bool(value))
+    monkeypatch.setattr(runner, "_BOOTSTRAP_STATE", types.SimpleNamespace(helper_promotion_callbacks=[], depth=1))
+
+    monkeypatch.setattr(runner, "BotRegistry", lambda *a, **k: object())
+    monkeypatch.setattr(runner, "DataBot", lambda *a, **k: object())
+    monkeypatch.setattr(runner, "create_context_builder", lambda *a, **k: object())
+    monkeypatch.setattr(runner, "SelfCodingEngine", lambda *a, **k: object())
+    monkeypatch.setattr(runner, "CodeDB", lambda *a, **k: object())
+    monkeypatch.setattr(runner, "MenaceMemoryManager", lambda *a, **k: object())
+
+    with pytest.raises(RuntimeError, match="requires an existing pipeline"):
+        runner.build_manager("ROI")
+
