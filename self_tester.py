from menace_sandbox.context_builder import create_context_builder
from menace_sandbox.weights import ensure_fresh_weights
from menace_sandbox.testing import _workflow_test_service_args, SelfTestService

builder = create_context_builder()
ensure_fresh_weights()

pytest_args, svc_kwargs = _workflow_test_service_args(builder)

svc = SelfTestService(**svc_kwargs)
results, passed_modules = svc.run_once(pytest_args=pytest_args)

print("Passes:", results.pass_count)
print("Fails:", results.fail_count)
print("Diagnostics:", results.diagnostics)
