from pathlib import Path
import json
from menace.clipped.profit_manager import ProfitManager
from menace.clipped.scoutboard import Scoutboard


def write_json(path: Path, data):
    with path.open('w', encoding='utf-8') as fh:
        json.dump(data, fh)


def test_cull_and_reassign(monkeypatch, tmp_path):
    clips = {
        '1': {'file_size_mb': 10.0, 'profit': 1.0, 'topic': 'old', 'confidence': 80.0, 'created': '2025-01-01T00:00:00'}
    }
    accounts = {'accounts': [{'id': 'a1', 'platform': 'yt', 'topics': ['old'], 'confidence': 80.0}]}
    topics = {'old': {'confidence': 80.0}}
    chamber = {}
    board = {'old': {'history': [{'profit_per_mb': 0.5}, {'profit_per_mb': 0.4}]}}
    write_json(tmp_path / 'clips.json', clips)
    write_json(tmp_path / 'accounts.json', accounts)
    write_json(tmp_path / 'clip_topics.json', topics)
    write_json(tmp_path / 'chamber.json', chamber)
    write_json(tmp_path / 'scoutboard.json', board)

    def fake_suggest(self, existing):
        return [{'name': 'new', 'trend_velocity': 1.5, 'similarity': 0.5, 'projected_profit_density': 1.0}]

    monkeypatch.setattr('menace.clipped.topic_prediction.TopicPredictionEngine.suggest_topics', fake_suggest)

    pm = ProfitManager(
        clips_file=tmp_path / 'clips.json',
        accounts_file=tmp_path / 'accounts.json',
        topics_file=tmp_path / 'clip_topics.json',
        chamber_file=tmp_path / 'chamber.json',
    )
    pm.scoutboard = Scoutboard(tmp_path / 'scoutboard.json')
    pm.run()

    topics_after = json.load(open(tmp_path / 'clip_topics.json'))
    accounts_after = json.load(open(tmp_path / 'accounts.json'))
    assert 'new' in topics_after
    assert topics_after['old'].get('hibernated')
    assert 'new' in accounts_after['accounts'][0]['topics']
