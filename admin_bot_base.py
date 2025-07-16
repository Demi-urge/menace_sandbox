from __future__ import annotations

import logging

from typing import Any, Final

from .database_router import DatabaseRouter, DBResult


logger = logging.getLogger(__name__)


class AdminBotBase:
    """Base class ensuring admin bots query all databases."""

    def __init__(
        self,
        db_router: DatabaseRouter | None = None,
        *,
        perform_health_check: bool = True,
    ) -> None:
        self.db_router: Final[DatabaseRouter] = db_router or DatabaseRouter()
        if perform_health_check:
            try:
                self.db_router.query_all(self.health_check_term())
            except Exception as exc:  # pragma: no cover - runtime failures
                logger.exception("Database health check failed: %s", exc)
                raise

    def health_check_term(self) -> str:
        """Return the query term used for health checks.

        Subclasses may override this to customise what term should be used
        when verifying database connectivity during initialisation.
        """
        return "admin:healthcheck"

    def query(self, term: str, **options: Any) -> DBResult:
        """Query all configured databases and return aggregated results.

        Parameters
        ----------
        term:
            Search term to query across databases.
        **options:
            Extra options forwarded to :meth:`DatabaseRouter.query_all`.

        Returns
        -------
        DBResult
            Aggregated search results from all databases.
        """
        if not self.db_router:
            raise RuntimeError("DatabaseRouter not configured")
        try:
            return self.db_router.query_all(term, **options)
        except Exception as exc:  # pragma: no cover - runtime failures
            logger.exception("Database query failed for term %s: %s", term, exc)
            raise
__all__ = ["AdminBotBase"]
