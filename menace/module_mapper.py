from __future__ import annotations

"""Compatibility wrapper exposing :mod:`module_mapper` in the ``menace`` package."""

from importlib import import_module

_base = import_module('module_mapper')

build_module_graph = _base.build_module_graph
cluster_modules = _base.cluster_modules
save_module_map = _base.save_module_map

__all__ = ['build_module_graph', 'cluster_modules', 'save_module_map']
