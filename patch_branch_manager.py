from __future__ import annotations

"""Manage patch branches based on confidence scores."""

from pathlib import Path
from datetime import datetime
import subprocess
import logging
import json

try:  # pragma: no cover - fallback for flat layout
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback
    from dynamic_path_router import resolve_path  # type: ignore

try:  # pragma: no cover - fallback for flat layout
    from .audit_trail import AuditTrail
except Exception:  # pragma: no cover - fallback
    from audit_trail import AuditTrail  # type: ignore

logger = logging.getLogger(__name__)


class PatchBranchManager:
    """Push patches to ``main`` or ``review/<patch_id>`` branches."""

    def __init__(
        self,
        repo: str | Path = ".",
        *,
        audit_trail: AuditTrail | None = None,
        main_branch: str = "main",
    ) -> None:
        self.repo = resolve_path(str(repo))
        self.audit_trail = audit_trail
        self.main_branch = main_branch

    # ------------------------------------------------------------------
    def finalize_patch(self, patch_id: str, score: float, threshold: float) -> str:
        """Push the current HEAD to a branch based on ``score``.

        Parameters
        ----------
        patch_id:
            Identifier for the patch.
        score:
            Confidence score for the patch.
        threshold:
            Minimum score required for automatic merge into ``main``.

        Returns
        -------
        str
            The branch name that received the commit.
        """

        branch = self.main_branch if score >= threshold else f"review/{patch_id}"
        action = "merged" if branch == self.main_branch else "review"
        try:
            subprocess.run(
                ["git", "push", "origin", f"HEAD:{branch}"],
                check=True,
                cwd=str(self.repo),
            )
        except Exception:
            logger.exception("git push failed for patch %s", patch_id)
            action = "failed"
        if self.audit_trail:
            try:
                payload = json.dumps(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "action": "patch_branch",
                        "patch_id": str(patch_id),
                        "branch": branch,
                        "score": float(score),
                        "result": action,
                    },
                    sort_keys=True,
                )
                self.audit_trail.record(payload)
            except Exception:
                logger.exception("audit trail logging failed")
        return branch


# ---------------------------------------------------------------------------

def finalize_patch_branch(
    patch_id: str,
    score: float,
    threshold: float,
    *,
    repo: str | Path = ".",
    audit_trail: AuditTrail | None = None,
    main_branch: str = "main",
) -> str:
    """Convenience wrapper around :class:`PatchBranchManager`."""

    manager = PatchBranchManager(repo, audit_trail=audit_trail, main_branch=main_branch)
    return manager.finalize_patch(patch_id, score, threshold)


__all__ = ["PatchBranchManager", "finalize_patch_branch"]
