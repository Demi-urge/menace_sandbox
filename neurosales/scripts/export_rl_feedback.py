import sys
from pathlib import Path
from dotenv import load_dotenv

from neurosales.rl_training import save_feedback_dataset

load_dotenv()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rl_feedback_export.json")
    save_feedback_dataset(str(path))
    print(f"Dataset saved to {path}")


if __name__ == "__main__":
    main()
