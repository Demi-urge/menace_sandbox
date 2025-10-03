import json

import pytest

from scripts import bootstrap_env


def test_parse_docker_log_envelope_handles_quoted_values():
    message = (
        'time="2024-05-16T20:10:40Z" level=warning msg="worker stalled; restarting" '
        'context="desktop-linux" restartCount=7 backoff="5s" error="context canceled"'
    )

    envelope = bootstrap_env._parse_docker_log_envelope(message)

    assert envelope["msg"] == "worker stalled; restarting"
    assert envelope["context"] == "desktop-linux"
    assert envelope["restartCount"] == "7"
    assert envelope["backoff"] == "5s"
    assert envelope["error"] == "context canceled"


def test_parse_docker_log_envelope_handles_json_structures():
    message = json.dumps(
        {
            "time": "2024-08-01T08:45:00Z",
            "level": "warning",
            "msg": "worker stalled; restarting",
            "details": {
                "context": "desktop-windows",
                "restartCount": 5,
                "metadata": {"lastError": "context canceled"},
            },
            "backoff": "45s",
        }
    )

    envelope = bootstrap_env._parse_docker_log_envelope(message)

    assert envelope["msg"] == "worker stalled; restarting"
    assert envelope["context"] == "desktop-windows"
    assert envelope["restartCount"] == "5"
    assert envelope["lastError"] == "context canceled"
    assert envelope["backoff"] == "45s"


def test_normalise_docker_warning_extracts_worker_metadata():
    message = (
        'time="2024-05-16T20:10:40Z" level=warning msg="worker stalled; restarting" '
        'context="desktop-linux" restartCount=7 backoff="5s" '
        'error="context canceled" lastRestart="2024-05-16T20:09:35Z"'
    )

    cleaned, metadata = bootstrap_env._normalise_docker_warning(message)

    assert "worker stalled" not in cleaned.lower()
    assert "restarts" in cleaned.lower()
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_context"] == "desktop-linux"
    assert metadata["docker_worker_restart_count"] == "7"
    assert metadata["docker_worker_backoff"] == "5s"
    assert metadata["docker_worker_last_error"] == "context canceled"
    assert metadata["docker_worker_last_restart"] == "2024-05-16T20:09:35Z"

    assert cleaned.startswith("Docker Desktop reported repeated restarts")
    assert "desktop-linux" in cleaned
    assert "5s" in cleaned


@pytest.mark.parametrize(
    "payload, expected_backoff",
    [
        (
            'WARNING: worker stalled; restarting in 10s\n{"Version":"26.1"}',
            "10s",
        ),
        (
            'worker stalled; restarting backoff=\"90s\"\n{"Version":"26.1"}',
            "90s",
        ),
    ],
)
def test_extract_json_document_normalises_worker_restart(payload, expected_backoff):
    json_fragment, warnings, metadata = bootstrap_env._extract_json_document(payload, "")

    assert json_fragment.strip().startswith("{")
    assert any("restarts" in warning.lower() for warning in warnings)
    assert metadata["docker_worker_health"] == "flapping"
    assert metadata["docker_worker_backoff"] == expected_backoff
