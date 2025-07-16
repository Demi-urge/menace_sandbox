import menace.dependency_verifier as dv


def test_verify_modules_missing():
    missing = dv.verify_modules(["nonexistent_package_xyz"])
    assert "nonexistent_package_xyz" in missing
