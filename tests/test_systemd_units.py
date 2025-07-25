import pathlib

import pytest

SERVICE_PATH = pathlib.Path('systemd/sandbox_purge.service')
TIMER_PATH = pathlib.Path('systemd/sandbox_purge.timer')


def test_systemd_files_exist():
    assert SERVICE_PATH.exists(), f"{SERVICE_PATH} missing"
    assert TIMER_PATH.exists(), f"{TIMER_PATH} missing"


def test_service_execstart():
    text = SERVICE_PATH.read_text()
    assert 'python -m sandbox_runner.cli --purge-stale' in text


def test_timer_references_service():
    text = TIMER_PATH.read_text()
    assert 'Unit=sandbox_purge.service' in text
