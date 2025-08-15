from importlib import import_module
_base = import_module('logging_utils')
setup_logging = _base.setup_logging
get_logger = _base.get_logger
log_record = _base.log_record
set_correlation_id = getattr(_base, 'set_correlation_id', lambda *_a, **_k: None)
__all__ = ['setup_logging', 'get_logger', 'log_record', 'set_correlation_id']
