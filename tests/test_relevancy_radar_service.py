import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

fake_qfe = types.ModuleType("quick_fix_engine")
fake_qfe.generate_patch = lambda path: 1
sys.modules["quick_fix_engine"] = fake_qfe

import menace_sandbox.module_retirement_service as module_retirement_service
import menace_sandbox.relevancy_radar as relevancy_radar
import menace_sandbox.relevancy_radar_service as relevancy_radar_service


def test_service_scan_updates_flags(monkeypatch, tmp_path):
    captured = {}

    def fake_update(flags):
        captured["flags"] = flags

    monkeypatch.setattr(relevancy_radar, "update_relevancy_metrics", fake_update)

    def fake_flag_unused_modules(mods, impact_stats=None):
        flags = {m: "retire" for m in mods}
        relevancy_radar.update_relevancy_metrics(flags)
        return flags

    monkeypatch.setattr(
        relevancy_radar.RelevancyRadar,
        "flag_unused_modules",
        staticmethod(fake_flag_unused_modules),
    )

    class DummyRetirementService:
        flags = None

        def __init__(self, root):
            pass

        def process_flags(self, flags):
            DummyRetirementService.flags = flags
            return {k: "retired" for k in flags}

    monkeypatch.setattr(
        module_retirement_service, "ModuleRetirementService", DummyRetirementService
    )

    service = relevancy_radar_service.RelevancyRadarService(tmp_path, interval=0)
    monkeypatch.setattr(service, "_modules", lambda: ["demo"])

    service._scan_once()

    assert service.flags() == {"demo": "retire"}
    assert captured["flags"] == {"demo": "retire"}
    assert DummyRetirementService.flags == {"demo": "retire"}
