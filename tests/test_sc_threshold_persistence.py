import yaml

import menace.data_bot as db
import menace.self_coding_thresholds as sct


def test_update_thresholds_persists_and_reloads(tmp_path, monkeypatch):
    cfg_path = tmp_path / "self_coding_thresholds.yaml"
    monkeypatch.setattr(sct, "_CONFIG_PATH", cfg_path)
    metrics = db.MetricsDB(path=tmp_path / "metrics.db")

    class DummyService:
        def __init__(self):
            self._thresholds = {}

        def update(
            self,
            bot,
            *,
            roi_drop=None,
            error_threshold=None,
            test_failure_threshold=None,
        ):
            pass

        def _publish(self, bot, rt):
            pass

    bot = db.DataBot(
        db=metrics,
        threshold_update_interval=0,
        threshold_service=DummyService(),
    )

    bot.update_thresholds(
        "alpha",
        roi_drop=-0.2,
        error_threshold=1.5,
        test_failure_threshold=0.3,
    )

    assert cfg_path.exists(), "thresholds should be written to config file"
    data = yaml.safe_load(cfg_path.read_text())
    assert data["bots"]["alpha"]["roi_drop"] == -0.2
    assert data["bots"]["alpha"]["error_increase"] == 1.5
    assert data["bots"]["alpha"]["test_failure_increase"] == 0.3

    bot.reload_thresholds("alpha")
    rt = bot.get_thresholds("alpha")
    assert rt.roi_drop == -0.2
    assert rt.error_threshold == 1.5
    assert rt.test_failure_threshold == 0.3
