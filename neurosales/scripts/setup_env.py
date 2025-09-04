import os
import subprocess
import sys
from pathlib import Path

from dynamic_path_router import resolve_path

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    root = Path(__file__).resolve().parents[1]
    reqs = root / "requirements.txt"
    print(f"Installing packages from {reqs}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(reqs)])

    heavy = resolve_path("neurosales/scripts/setup_heavy_deps.py")
    print("Running heavy dependency setup...")
    subprocess.check_call([sys.executable, str(heavy)])

    db_url = os.getenv("NEURO_DB_URL")
    if db_url:
        print("Running database migrations...")
        from scripts import migrate
        migrate.main()
    else:
        print("NEURO_DB_URL not set; skipping migrations")

    print("Environment setup complete")


if __name__ == "__main__":
    main()
