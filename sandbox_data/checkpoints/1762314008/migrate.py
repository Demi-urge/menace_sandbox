import os
import argparse
from pathlib import Path
from alembic import command
from alembic.config import Config
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run alembic migrations")
    parser.add_argument("action", choices=["upgrade", "downgrade"], nargs="?", default="upgrade")
    parser.add_argument("revision", nargs="?", default="head")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = Config()
    config.set_main_option("script_location", str(root / "alembic"))

    db_url = os.getenv("NEURO_DB_URL")
    if not db_url:
        raise RuntimeError("NEURO_DB_URL environment variable not set")
    config.set_main_option("sqlalchemy.url", db_url)

    if args.action == "upgrade":
        command.upgrade(config, args.revision)
    else:
        command.downgrade(config, args.revision)


if __name__ == "__main__":
    main()
