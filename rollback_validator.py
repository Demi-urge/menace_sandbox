from __future__ import annotations

"""Additional verification for distributed rollbacks."""

import logging
from typing import Iterable

from .rollback_manager import RollbackManager


class RollbackValidator(RollbackManager):
    """Verify rollback success across nodes."""

    def verify_rollback(self, patch_id: str, nodes: Iterable[str]) -> bool:
        """Check each node for rollback confirmation."""
        confirmed = 0
        node_list = list(nodes)
        for node in node_list:
            try:
                patches = self.applied_patches(node)
                if patch_id not in {p.patch_id for p in patches}:
                    confirmed += 1
            except Exception:
                self.logger.exception("verification failed for %s", node)
        quorum = len(node_list) // 2 + 1
        return confirmed >= quorum

    def rollback_and_verify(self, patch_id: str, nodes: Iterable[str]) -> bool:
        self.rollback(patch_id)
        return self.verify_rollback(patch_id, nodes)


__all__ = ["RollbackValidator"]
