from deployment_governance import DeploymentGovernor
from foresight_gate import is_foresight_safe_to_promote
from borderline_bucket import BorderlineBucket
from foresight_tracker import ForesightTracker
from workflow_graph import WorkflowGraph

tracker = ForesightTracker()
graph = WorkflowGraph()
governor = DeploymentGovernor()
bucket = BorderlineBucket()

patch = ["tune_step_a"]

decision = is_foresight_safe_to_promote("workflow-1", patch, tracker, graph)

if decision.safe:
    result = governor.evaluate(
        {},
        alignment_status="pass",
        raroi=1.2,
        confidence=0.8,
        sandbox_roi=0.3,
        adapter_roi=1.1,
        policy=None,
    )
else:
    bucket.enqueue(
        "workflow-1",
        raroi=0.8,
        confidence=decision.forecast.get("confidence"),
        context=decision.reasons,
    )
    result = {"verdict": "pilot", "reasons": decision.reasons}

print(result)
