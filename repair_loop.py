repair_limit = 3
attempt = 1
diag = results.diagnostics[0]
real_path = diag["file"]
test_name = diag["test_name"]
description = f"Fix: {diag['error_summary']} in {diag['file']}"

while attempt <= repair_limit:
    print(f"\n🔁 Repair attempt {attempt}...")

    flags, context_meta = quick_fix.validate_patch(
        module_path=real_path,
        description=description,
        repo_root=None,
        provenance_token=provenance
    )
    if flags:
        raise RuntimeError(f"Repair validation failed (attempt {attempt}): {flags}")

    context_meta["repair_attempt"] = attempt

    quick_fix.apply_validated_patch(
        module_path=real_path,
        flags=flags,
        context_meta=context_meta,
        repo_root=None,
        provenance_token=provenance
    )

    # Scoped test retry
    results, passed_modules = svc.run_once(pytest_args=["-k", test_name])
    if results.fail_count == 0:
        print("✅ Tests passed after repair.")
        break

    attempt += 1

else:
    raise RuntimeError(f"❌ Repair loop exhausted after {repair_limit} attempts.")
