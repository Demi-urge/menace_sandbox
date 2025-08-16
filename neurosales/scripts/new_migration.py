import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from alembic import command
from alembic.config import Config

load_dotenv()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/new_migration.py \"message\"")
        sys.exit(1)
    message = sys.argv[1]

    root = Path(__file__).resolve().parents[1]
    config = Config()
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", os.getenv("NEURO_DB_URL", "sqlite://"))

    command.revision(config, message=message, autogenerate=True)


if __name__ == "__main__":
    main()
