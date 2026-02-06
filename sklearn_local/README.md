# Local sklearn shim

This package is intentionally **not** named `sklearn` because that name is reserved
for the real scikit-learn dependency. Keeping this shim under a different name
prevents it from shadowing the external library on `sys.path`.
