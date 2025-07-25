import pathlib

import pytest

PURGE_SERVICE = pathlib.Path('systemd/sandbox_purge.service')
PURGE_TIMER = pathlib.Path('systemd/sandbox_purge.timer')
CLEANUP_SERVICE = pathlib.Path('systemd/cleanup.service')
CLEANUP_TIMER = pathlib.Path('systemd/cleanup.timer')


def test_systemd_files_exist():
    for path in [PURGE_SERVICE, PURGE_TIMER, CLEANUP_SERVICE, CLEANUP_TIMER]:
        assert path.exists(), f"{path} missing"


def test_service_execstart():
    text = PURGE_SERVICE.read_text()
    assert 'python -m sandbox_runner.cli --purge-stale' in text
    text = CLEANUP_SERVICE.read_text()
    assert 'python -m sandbox_runner.cli cleanup' in text


def test_timer_references_service():
    text = PURGE_TIMER.read_text()
    assert 'Unit=sandbox_purge.service' in text
    text = CLEANUP_TIMER.read_text()
    assert 'Unit=cleanup.service' in text
