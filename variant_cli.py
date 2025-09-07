from __future__ import annotations

"""Command-line interface for spawning and evaluating variant branches."""

import argparse
import json
from typing import Any, Dict

from vector_service.context_builder import ContextBuilder

from .experiment_manager import ExperimentManager
from .variant_manager import VariantManager
from .mutation_lineage import MutationLineage
from .data_bot import DataBot
from .capital_management_bot import CapitalManagementBot


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Spawn and test a variant")
    parser.add_argument("patch_id", type=int, help="Parent patch or event id")
    parser.add_argument("variant", help="Name for the new variant")
    args = parser.parse_args()

    lineage = MutationLineage()
    # Clone the branch explicitly so operators can see the new patch id
    new_patch = lineage.clone_branch_for_ab_test(args.patch_id, args.variant)

    builder = ContextBuilder()
    exp_mgr = ExperimentManager(
        DataBot(),
        CapitalManagementBot(),
        context_builder=builder,
        lineage=lineage,
    )
    vm = VariantManager(exp_mgr)
    event_id, results = vm.ab_test_branch(args.patch_id, args.variant)
    comparisons = exp_mgr.compare_variants(results)

    payload: Dict[str, Any] = {
        "new_patch_id": new_patch,
        "event_id": event_id,
        "results": [r.__dict__ for r in results],
        "comparisons": {
            f"{a} vs {b}": {"t_stat": t, "p_value": p}
            for (a, b), (t, p) in comparisons.items()
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
