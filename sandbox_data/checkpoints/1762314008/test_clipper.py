import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.clipped.clipper as clipper
from pathlib import Path


def test_clipper_cli_defaults(monkeypatch):
    called = {}
    def fake_process_inputs(input_dir, output_dir):
        called['input'] = input_dir
        called['output'] = output_dir
    monkeypatch.setattr(clipper, 'process_inputs', fake_process_inputs)
    clipper.cli([])
    assert called['input'] == Path('videos')
    assert called['output'] == Path('output_clips')


def test_process_inputs_creates_dirs(tmp_path):
    input_dir = tmp_path / 'v'
    output_dir = tmp_path / 'o'
    clipper.process_inputs(input_dir, output_dir)
    assert input_dir.exists()
    assert output_dir.exists()
