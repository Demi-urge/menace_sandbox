import json
from pathlib import Path
import menace.clipped.topic_classifier as tc


def test_classifier_cli_defaults(monkeypatch):
    called = {}
    def fake_classify(input_dir, topics_file):
        called['input'] = input_dir
        called['topics'] = topics_file
    monkeypatch.setattr(tc, 'classify_clips', fake_classify)
    tc.cli([])
    assert called['input'] == 'output_clips'
    assert called['topics'] == 'clip_topics.json'


def test_classify_clips_creates_dir(tmp_path, monkeypatch):
    tfile = tmp_path / 'topics.json'
    tfile.write_text(json.dumps({'topics': []}))
    in_dir = tmp_path / 'clips'
    processed = []
    monkeypatch.setattr(tc, 'process_clip', lambda path, topics: processed.append(path))
    tc.classify_clips(str(in_dir), str(tfile))
    assert in_dir.exists()
    assert processed == []
