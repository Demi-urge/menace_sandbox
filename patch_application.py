from menace_sandbox.context_builder import create_context_builder
from menace_sandbox.quick_fix_engine import quick_fix
import pathlib
from types import SimpleNamespace

repo_root = pathlib.Path(__file__).parent.resolve()
builder = create_context_builder(repo_root=repo_root)
provenance = builder.provenance_token

# Provide a lightweight manager stub so ``quick_fix.validate_patch`` receives
# both the ``SelfCodingManager``-like object and the bound context builder it
# expects. The real manager is much heavier to construct, but the stub is
# sufficient for validation and patch application flows that only need a
# context-aware object.
manager = SimpleNamespace(context_builder=builder)

module_path = str(repo_root / "menace_sandbox/modules/YOUR_MODULE.py")
description = "Fix: patch logic for XYZ"  # customize this

valid, flags = quick_fix.validate_patch(
    module_path=module_path,
    description=description,
    repo_root=repo_root,
    provenance_token=provenance,
    manager=manager,
    context_builder=builder,
)

if "missing_context" in flags:
    raise RuntimeError(
        "Patch validation requires both a SelfCodingManager and ContextBuilder. "
        "Provide a configured manager (with the builder attached) before running"
    )

if valid:
    quick_fix.apply_validated_patch(
        module_path=module_path,
        flags=flags,
        context_meta={"description": description},
        repo_root=repo_root,
        provenance_token=provenance,
        manager=manager,
        context_builder=builder,
    )
    print("✅ Patch successfully applied to", module_path)
else:
    print("[!] Patch validation failed. Flags:", flags)
