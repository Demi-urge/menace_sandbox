import argparse
import subprocess
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

from neurosales.dynamic_harvest import ensure_playwright_browsers

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup heavy dependencies")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download embedding weights without installing packages",
    )
    args = parser.parse_args()

    if not args.download_only:
        root = Path(__file__).resolve().parents[1]
        reqs = root / "requirements-extra.txt"
        print(f"Installing packages from {reqs}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(reqs)]
        )

        if shutil.which("playwright"):
            print("Installing Playwright browsers...")
            subprocess.call(["playwright", "install", "--with-deps"])
            if ensure_playwright_browsers():
                print("Playwright installation verified")
            else:
                print(
                    "WARNING: Playwright browsers missing. Run 'playwright install --with-deps'"
                )
        else:
            print("Playwright not found; skipping browser install")

    try:
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import login
        import os

        login(token=os.getenv("HUGGINGFACE_API_TOKEN"))
        print("Prefetching embedding model weights...")
        SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Could not load SentenceTransformer: {exc}")

    print("Heavy dependencies are ready")


if __name__ == "__main__":
    main()
