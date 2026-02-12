from menace.self_coding_divergence_detector import load_divergence_detector_config


def test_load_divergence_detector_config_parses_missing_metric_controls(tmp_path):
    config_path = tmp_path / "guard.yaml"
    config_path.write_text(
        """
self_coding_divergence_guard:
  fail_closed_on_missing_metrics: false
  missing_metric_pause_cycles: 5
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_divergence_detector_config(str(config_path))

    assert cfg.fail_closed_on_missing_metrics is False
    assert cfg.missing_metric_pause_cycles == 5


def test_load_divergence_detector_config_clamps_missing_metric_cycles(tmp_path):
    config_path = tmp_path / "guard.yaml"
    config_path.write_text(
        """
self_coding_divergence_guard:
  missing_metric_pause_cycles: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_divergence_detector_config(str(config_path))

    assert cfg.missing_metric_pause_cycles == 1
