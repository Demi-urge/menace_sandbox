from menace_sandbox.context_builder import create_context_builder
from menace_sandbox.quick_fix_engine import quick_fix
import pathlib

repo_root = pathlib.Path(__file__).parent.resolve()
builder = create_context_builder(repo_root=repo_root)
provenance = builder.provenance_token

module_path = str(repo_root / "menace_sandbox/modules/YOUR_MODULE.py")
description = "Fix: patch logic for XYZ"  # customize this

valid, flags = quick_fix.validate_patch(
    module_path=module_path,
    description=description,
    repo_root=repo_root,
    provenance_token=provenance,
    context_builder=builder,
)

if "missing_context" in flags:
    raise RuntimeError("Context builder was missing — patch flow is invalid")

if valid:
    quick_fix.apply_validated_patch(
        module_path=module_path,
        flags=flags,
        context_meta={"description": description},
        repo_root=repo_root,
        provenance_token=provenance,
        context_builder=builder,
    )
    print("✅ Patch successfully applied to", module_path)
else:
    print("[!] Patch validation failed. Flags:", flags)
