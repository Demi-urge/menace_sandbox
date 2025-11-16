"""Local Stripe policy fallbacks for offline sandbox execution.

Production Menace deployments source policy documents from a remote
configuration service. When running locally we only need a concise statement
that mirrors the intent of the upstream policy so dependent modules continue to
work. The helpers below expose a structured representation that callers can use
for richer messaging while still offering the simple ``PAYMENT_ROUTER_NOTICE``
string expected by legacy modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

__all__ = ["PolicyClause", "PAYMENT_ROUTER_NOTICE", "iter_policy_clauses"]


@dataclass(frozen=True, slots=True)
class PolicyClause:
    """Represents a single policy clause for display or logging."""

    heading: str
    body: str

    def as_bullet(self) -> str:
        """Return the clause formatted as a Markdown bullet."""

        return f"- **{self.heading}:** {self.body}"


_POLICY_CLAUSES: tuple[PolicyClause, ...] = (
    PolicyClause(
        "Centralised routing",
        "All billing activity must flow through ``stripe_billing_router`` so that "
        "API keys are never embedded directly in tools or prompts.",
    ),
    PolicyClause(
        "Ledger integration",
        "Each processed charge must be recorded via ``billing_logger`` and "
        "``stripe_ledger`` to ensure accurate financial reconciliation.",
    ),
    PolicyClause(
        "Failure handling",
        "Missing billing hooks or bypassing the router is treated as a critical "
        "error and must abort the operation.",
    ),
)

PAYMENT_ROUTER_NOTICE = "\n".join(clause.as_bullet() for clause in _POLICY_CLAUSES)


def iter_policy_clauses() -> Iterator[PolicyClause]:
    """Yield the static policy clauses.

    Returning an iterator keeps the API similar to the remote-backed version
    used in production. Consumers can convert the iterator to a list when
    needed.
    """

    return iter(_POLICY_CLAUSES)
