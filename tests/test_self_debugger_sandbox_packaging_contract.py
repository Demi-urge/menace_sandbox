from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap


def test_menace_code_database_shim_exports_packaging_contract(monkeypatch) -> None:
    """The packaged shim must expose both public and compatibility-private symbols."""
    for module_name in ("menace.code_database", "menace", "code_database"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    module = importlib.import_module("menace.code_database")

    assert hasattr(module, "PatchHistoryDB")
    assert hasattr(module, "_hash_code")


def test_packaged_self_debugger_sandbox_import_allows_hash_code_contract() -> None:
    """Packaged import should succeed without ImportError related to _hash_code."""
    script = textwrap.dedent(
        """
        import importlib
        import os
        import sys
        import types

        os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

        for module_name in ("menace", "menace.code_database", "code_database"):
            sys.modules.pop(module_name, None)

        def _stub_module(name: str, **attrs):
            module = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(module, key, value)
            sys.modules.setdefault(name, module)

        _stub_module("menace.logging_utils", log_record=lambda *a, **k: None)
        _stub_module("menace.retry_utils", with_retry=lambda func, *a, **k: func)
        _stub_module("menace.error_logger", ErrorLogger=object, TelemetryEvent=object)
        _stub_module("menace.target_region", TargetRegion=object, extract_target_region=lambda *a, **k: None)
        _stub_module("menace.knowledge_graph", KnowledgeGraph=object)
        _stub_module("menace.human_alignment_agent", HumanAlignmentAgent=object)
        _stub_module("menace.human_alignment_flagger", _collect_diff_data=lambda *a, **k: {})
        _stub_module("menace.violation_logger", log_violation=lambda *a, **k: None)
        _stub_module("menace.sandbox_runner.scoring", record_run=lambda *a, **k: None)
        _stub_module(
            "menace.db_router",
            GLOBAL_ROUTER=types.SimpleNamespace(get_connection=lambda *a, **k: None),
            init_db_router=lambda *a, **k: None,
        )
        _stub_module("menace.automated_debugger", AutomatedDebugger=object)
        _stub_module("menace.self_coding_engine", SelfCodingEngine=object)
        _stub_module("menace.audit_trail", AuditTrail=object)
        _stub_module("menace.patch_attempt_tracker", PatchAttemptTracker=object)
        _stub_module("menace.self_improvement_policy", SelfImprovementPolicy=object)
        _stub_module("menace.error_cluster_predictor", ErrorClusterPredictor=object)
        _stub_module("menace.dynamic_path_router", resolve_path=lambda p: p)

        module = importlib.import_module("menace.self_debugger_sandbox")
        assert hasattr(module, "SelfDebuggerSandbox")
        print("IMPORT_OK")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd="/workspace/menace_sandbox",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout
