from __future__ import annotations

import os

import bootstrap_timeout_policy as btp


def test_component_budgets_expand_with_overruns(monkeypatch):
    floors = {
        "vectorizers": 120.0,
        "retrievers": 90.0,
        "db_indexes": 60.0,
        "orchestrator_state": 50.0,
    }

    monkeypatch.setattr(btp, "_state_host_key", lambda: "test-host")
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    monkeypatch.setattr(btp, "_save_timeout_state", lambda state: None)
    monkeypatch.setattr(
        btp,
        "_load_timeout_state",
        lambda: {
            "test-host": {
                "component_overruns": {
                    "retrievers": {
                        "overruns": 3,
                        "max_elapsed": 180.0,
                        "expected_floor": 120.0,
                        "suggested_floor": 200.0,
                    }
                }
            }
        },
    )

    telemetry = {
        "shared_timeout": {
            "timeline": [
                {
                    "label": "vectorizer warmup",
                    "elapsed": 150.0,
                    "effective": 100.0,
                }
            ]
        }
    }

    budgets = btp.compute_prepare_pipeline_component_budgets(
        component_floors=floors, telemetry=telemetry, load_average=2.0
    )

    assert budgets["vectorizers"] > floors["vectorizers"]
    assert budgets["retrievers"] > floors["retrievers"]

