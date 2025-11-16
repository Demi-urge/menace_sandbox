"""Base class providing DB router access for admin bots."""

import logging
from typing import Any, Final

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router

logger = logging.getLogger(__name__)


class AdminBotBase:
    """Ensure admin bots have an initialised :class:`DBRouter`."""

    def __init__(
        self,
        db_router: DBRouter | None = None,
        *,
        perform_health_check: bool = True,
    ) -> None:
        self.db_router: Final[DBRouter] = db_router or GLOBAL_ROUTER or init_db_router("admin")
        if perform_health_check:
            try:
                conn = self.db_router.get_connection("bots")
                conn.execute("SELECT 1")
            except Exception as exc:  # pragma: no cover - runtime failures
                logger.exception("Database health check failed: %s", exc)
                raise

    def health_check_term(self) -> str:
        """Return a dummy term for legacy compatibility."""

        return "admin:healthcheck"

    def query(self, term: str, **options: Any) -> None:
        """Deprecated; legacy method retained for compatibility."""

        raise NotImplementedError(
            "AdminBotBase.query has been removed; query databases directly"
        )


__all__ = ["AdminBotBase"]
