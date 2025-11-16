import sys
from pathlib import Path
from dotenv import load_dotenv

from neurosales.engagement_dataset import collect_engagement_logs

load_dotenv()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("engagement_data.csv")
    collect_engagement_logs(str(path))
    print(f"Dataset saved to {path}")


if __name__ == "__main__":
    main()
