import sys
from pathlib import Path
from dotenv import load_dotenv

from neurosales.rl_training import save_feedback_dataset, train_models

load_dotenv()


def main() -> None:
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rl_feedback_export.json")
    save_feedback_dataset(str(dataset))
    train_models(str(dataset))
    print("Models updated from feedback")


if __name__ == "__main__":
    main()
