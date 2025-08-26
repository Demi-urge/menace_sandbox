from __future__ import annotations

import warnings

from audit import log_db_access as _log_db_access


def log_db_access(*args: object, **kwargs: object) -> None:
    """Deprecated wrapper for :func:`audit.log_db_access`.

    This function forwards all arguments to :func:`audit.log_db_access`. It will
    be removed in a future release; callers should import ``log_db_access``
    directly from :mod:`audit`.
    """
    warnings.warn(
        "audit_db_access.log_db_access is deprecated; use audit.log_db_access",
        DeprecationWarning,
        stacklevel=2,
    )
    _log_db_access(*args, **kwargs)


__all__ = ["log_db_access"]
