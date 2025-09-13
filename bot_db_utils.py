from __future__ import annotations

import inspect
import logging
import types
from functools import wraps
from typing import Callable, Optional

from .unified_event_bus import UnifiedEventBus

from .db_router import DBRouter
from .bot_registry import BotRegistry


def wrap_bot_methods(
    bot: object,
    db_router: DBRouter,
    bot_registry: Optional[BotRegistry] = None,
    *,
    event_bus: Optional[UnifiedEventBus] = None,
) -> None:
    """Attach ``db_router`` and optional helpers to ``bot`` and instrument callable attributes."""

    logger = logging.getLogger(__name__)
    setattr(bot, "db_router", db_router)
    if bot_registry:
        setattr(bot, "bot_registry", bot_registry)
    if event_bus:
        setattr(bot, "event_bus", event_bus)

    for name, attr in inspect.getmembers(bot):
        if name.startswith("_") or not callable(attr):
            continue

        func = attr.__func__ if isinstance(attr, types.MethodType) else attr

        @wraps(func)
        def wrapper(*args, __f=func, __name=name, **kwargs):
            try:
                db_router.get_connection("bots").execute("SELECT 1")
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.warning("DB query for %s failed: %s", __name, exc)

            from_name = getattr(bot, "name", bot.__class__.__name__)
            eb = event_bus or getattr(bot, "event_bus", None)

            if bot_registry:
                try:
                    manager_obj = getattr(bot, "manager", None)
                    d_bot = getattr(bot, "data_bot", None)
                    bot_registry.register_bot(
                        from_name,
                        manager=manager_obj,
                        data_bot=d_bot,
                        is_coding_bot=bool(manager_obj and d_bot),
                    )
                except Exception as exc:
                    logger.warning(
                        "bot registration failed for %s: %s",
                        from_name,
                        exc,
                        exc_info=True,
                    )

            call_args = args[1:] if args and args[0] is bot else args
            for a in call_args:
                if not hasattr(a, "name"):
                    continue
                to_name = getattr(a, "name")
                try:
                    hash(to_name)
                except Exception:
                    logger.debug("skipping non-hashable name: %r", to_name)
                    continue
                if bot_registry:
                    try:
                        bot_registry.register_interaction(from_name, to_name)
                    except Exception as exc:
                        logger.warning(
                            "interaction registration from %s to %s failed: %s",
                            from_name,
                            to_name,
                            exc,
                            exc_info=True,
                        )
                if eb:
                    try:
                        eb.publish("bot:call", {"from": from_name, "to": to_name})
                    except Exception as exc:
                        logger.warning(
                            "event publish failed for %s -> %s: %s",
                            from_name,
                            to_name,
                            exc,
                            exc_info=True,
                        )

            return __f(*args, **kwargs)

        bound = (
            wrapper.__get__(bot, bot.__class__)
            if isinstance(attr, types.MethodType)
            else wrapper
        )
        setattr(bot, name, bound)
