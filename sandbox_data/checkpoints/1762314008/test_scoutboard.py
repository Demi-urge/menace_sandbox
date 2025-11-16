from pathlib import Path
import json
from menace.clipped.profit_manager import ProfitManager


def write_json(path: Path, data):
    with path.open('w', encoding='utf-8') as fh:
        json.dump(data, fh)


def test_scoutboard_updates(tmp_path):
    clips = {
        '1': {'file_size_mb': 10.0, 'profit': 5.0, 'topic': 'old', 'confidence': 80.0, 'created': '2025-01-01T00:00:00'}
    }
    accounts = {'accounts': [{'id': 'a1', 'platform': 'yt', 'topics': ['old'], 'confidence': 80.0}]}
    topics = {'old': {'confidence': 80.0}}
    chamber = {}
    write_json(tmp_path / 'clips.json', clips)
    write_json(tmp_path / 'accounts.json', accounts)
    write_json(tmp_path / 'clip_topics.json', topics)
    write_json(tmp_path / 'chamber.json', chamber)
    scout_path = tmp_path / 'scoutboard.json'
    write_json(scout_path, {})

    pm = ProfitManager(
        clips_file=tmp_path / 'clips.json',
        accounts_file=tmp_path / 'accounts.json',
        topics_file=tmp_path / 'clip_topics.json',
        chamber_file=tmp_path / 'chamber.json',
    )
    pm.scoutboard.path = scout_path
    pm.run()
    board = json.load(open(scout_path))
    assert 'old' in board
    assert 'history' in board['old']
