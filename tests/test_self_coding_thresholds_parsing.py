"""Parsing resilience for ``self_coding_thresholds`` configuration."""

from pathlib import Path

import menace_sandbox.self_coding_thresholds as sct


def test_load_config_ignores_empty_key_lines(tmp_path: Path) -> None:
    cfg = tmp_path / "self_coding_thresholds.yaml"
    cfg.write_text(
        "\n".join(
            [
                "bots:",
                "  GoodBot:",
                "    roi_drop: -0.1",
                ": {}",  # malformed empty key that previously broke parsing
                "  AnotherBot:",
                "    error_increase: 2.0",
                "",
            ]
        )
    )

    result = sct._load_config(cfg)

    assert result["bots"]["GoodBot"]["roi_drop"] == -0.1
    assert result["bots"]["AnotherBot"]["error_increase"] == 2.0
