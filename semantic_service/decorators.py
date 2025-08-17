"""Compatibility wrappers for :mod:`vector_service.decorators`.

Re-export decorators and metric gauges from :mod:`vector_service` with
warnings. Attribute assignments also propagate to the underlying module so
monkeypatching works as expected. This module will be removed once callers
switch to ``vector_service``.
"""

import warnings
import sys
import types
import vector_service.decorators as _dec

class _Proxy(types.ModuleType):
    def __setattr__(self, name, value):  # pragma: no cover - simple forwarding
        setattr(_dec, name, value)
        super().__setattr__(name, value)

module = sys.modules[__name__]
module.__class__ = _Proxy
module.log_and_measure = _dec.log_and_measure
module.log_and_time = _dec.log_and_time
module.track_metrics = _dec.track_metrics
module._CALL_COUNT = _dec._CALL_COUNT
module._LATENCY_GAUGE = _dec._LATENCY_GAUGE
module._RESULT_SIZE_GAUGE = _dec._RESULT_SIZE_GAUGE

__all__ = ["log_and_measure", "log_and_time", "track_metrics"]

warnings.warn(
    "`semantic_service.decorators` is deprecated; use `vector_service.decorators`",
    DeprecationWarning,
    stacklevel=2,
)
