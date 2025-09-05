import csv
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))
from dynamic_path_router import resolve_path
from neurosales.policy_learning import PolicyLearner

load_dotenv()


def _load_records(path: Path):
    """Return a list of state/action/reward dicts from ``path``."""
    if path.suffix in {".jsonl", ".log"}:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _infer_spec(records):
    actions = sorted({r["action"] for r in records})
    first = records[0]
    if isinstance(first["state"], list):
        state_dim = len(first["state"])
    else:
        state_dim = len([k for k in first.keys() if k not in {"action", "reward"}])
    return actions, state_dim


def main() -> None:
    root = resolve_path(".")
    dataset = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else resolve_path("neurosales/tests/data/policy_train.json")
    )
    records = _load_records(dataset)
    actions, state_dim = _infer_spec(records)
    tactics = {a: "flat" for a in actions}
    learner = PolicyLearner(actions, tactics, state_dim=state_dim)
    tmp_path = root / "_tmp_policy_data.json"
    with open(tmp_path, "w") as f:
        json.dump(records, f)
    learner.train_from_dataset(str(tmp_path))
    tmp_path.unlink()
    weights_path = resolve_path("neurosales") / "policy_params.json"
    with open(weights_path, "w") as f:
        json.dump(learner.brain.params, f)
    print(f"Weights saved to {weights_path}")


if __name__ == "__main__":
    main()
