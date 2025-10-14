"""Tests for the lightweight YAML fallback implementation."""

from __future__ import annotations

import builtins
import importlib
import sys
from contextlib import contextmanager

import pytest

import yaml_fallback


@contextmanager
def _simulate_missing_pyyaml():
    """Temporarily ensure :mod:`PyYAML` is unavailable for the test duration."""

    original_module = sys.modules.pop("yaml", None)
    real_import = builtins.__import__

    def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "yaml" or (name.startswith("yaml.") and level == 0):
            raise ModuleNotFoundError("No module named 'yaml'")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = _failing_import
    try:
        importlib.reload(yaml_fallback)
        yield yaml_fallback
    finally:
        builtins.__import__ = real_import
        if original_module is not None:
            sys.modules["yaml"] = original_module
        else:
            sys.modules.pop("yaml", None)
        importlib.reload(yaml_fallback)


def test_safe_load_supports_nested_documents():
    with _simulate_missing_pyyaml() as fallback:
        yaml = fallback.get_yaml("unit-test", warn=False)
        document = yaml.safe_load(
            """
            paths:
              data: /tmp/data
              logs: /tmp/logs
            features:
              - name: tracing
                enabled: true
              - name: metrics
                thresholds:
                  error: 0.5
                  warn: 0.2
            """
        )

    assert document == {
        "paths": {"data": "/tmp/data", "logs": "/tmp/logs"},
        "features": [
            {"name": "tracing", "enabled": True},
            {"name": "metrics", "thresholds": {"error": 0.5, "warn": 0.2}},
        ],
    }


def test_block_scalar_round_trip():
    config = "message: |\n  Hello sandbox!\n  Keep calm.\n"
    with _simulate_missing_pyyaml() as fallback:
        yaml = fallback.get_yaml("unit-test", warn=False)
        parsed = yaml.safe_load(config)
        dumped = yaml.safe_dump(parsed)
        assert yaml.safe_load(dumped) == parsed

    assert parsed == {"message": "Hello sandbox!\nKeep calm.\n"}


def test_invalid_indentation_raises_parse_error():
    with _simulate_missing_pyyaml() as fallback:
        yaml = fallback.get_yaml("unit-test", warn=False)
        with pytest.raises(yaml_fallback.YAMLParseError):
            yaml.safe_load("key:\n value")

