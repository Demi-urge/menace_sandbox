from menace_sandbox.quick_fix_engine import quick_fix
from menace_sandbox.context_builder import create_context_builder

# Clone path created by bootstrap script
repo_root = "/tmp/menace_sandbox_clone"
module_path = f"{repo_root}/menace_sandbox/modules/xyz.py"
description = "Fix: update function signature for new call pattern"

# Get provenance from clone-based context
builder = create_context_builder(repo_root=repo_root)
provenance = builder.provenance_token

# Validate (flags remaining = fail)
flags, context_meta = quick_fix.validate_patch(
    module_path=module_path,
    description=description,
    repo_root=repo_root,
    provenance_token=provenance
)
if flags:
    raise RuntimeError(f"Validation failed: flags present: {flags}")

# Apply patch
quick_fix.apply_validated_patch(
    module_path=module_path,
    flags=flags,
    context_meta=context_meta,
    repo_root=repo_root,
    provenance_token=provenance
)
