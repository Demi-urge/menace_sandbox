from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from typing import Callable, Mapping, Any
import inspect
import logging

logger = logging.getLogger(__name__)


def auto_link(mapping: Mapping[str, str]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to automatically link inserted rows.

    ``mapping`` maps kwarg names to the instance method used to create the
    link. Each kwarg should be an ``Iterable`` of ids. The decorated function
    must return the id of the newly inserted row.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            link_args = {k: kwargs.pop(k, None) for k in mapping}
            src_menace = kwargs.get("source_menace_id")
            record_id = func(self, *args, **kwargs)
            for arg, method_name in mapping.items():
                values = link_args.get(arg)
                if not values:
                    continue
                if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
                    values = [values]
                link_fn = getattr(self, method_name, None)
                if not link_fn:
                    continue
                for val in values:
                    try:
                        params: dict[str, Any] = {}
                        if (
                            src_menace is not None
                            and "source_menace_id" in inspect.signature(link_fn).parameters
                        ):
                            params["source_menace_id"] = src_menace
                        link_fn(record_id, val, **params)
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.error("auto link failed", exc_info=True)
            return record_id

        return wrapper

    return decorator


__all__ = ["auto_link"]
