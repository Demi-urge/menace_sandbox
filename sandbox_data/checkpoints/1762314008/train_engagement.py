import sys
import csv
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))
from dynamic_path_router import resolve_path

import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump

load_dotenv()


def load_data(path: Path):
    X = []
    y = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row["length"]), float(row["exclam"]), float(row["question"])])
            y.append(float(row["engagement"]))
    return np.array(X, dtype=float), np.array(y, dtype=float)


def main() -> None:
    default_path = resolve_path("neurosales/tests/data/engagement_train.csv")
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    model_path = resolve_path("neurosales") / "engagement_model.joblib"

    X, y = load_data(data_path)
    model = LinearRegression()
    model.fit(X, y)
    dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
