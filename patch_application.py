from menace_sandbox.quick_fix_engine import quick_fix
from menace_sandbox.context_builder import create_context_builder
import pathlib

# Assuming you're working in the main repo, set this to your actual repo root
repo_root = pathlib.Path(__file__).parent.resolve()
builder = create_context_builder(repo_root=repo_root)
provenance = builder.provenance_token

# Update these values for the module you're patching
module_path = "menace_sandbox/modules/your_target_module.py"
description = "Fix: resolve issue in ..."

flags, context_meta = quick_fix.validate_patch(
    module_path=module_path,  # Replace with real module
    description=description,  # Your patch description
    repo_root=repo_root,
    provenance_token=provenance,
    context_builder=builder,
)

if not flags:
    quick_fix.apply_validated_patch(
        module_path=module_path,
        description=description,
        repo_root=repo_root,
        provenance_token=provenance,
        context_meta=context_meta,
        context_builder=builder,
    )
else:
    print("Patch validation failed. Flags:", flags)
