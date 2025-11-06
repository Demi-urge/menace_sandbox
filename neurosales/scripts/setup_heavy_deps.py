import argparse
import importlib
import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
import typing
try:  # pragma: no cover - optional dependency in minimal envs
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - degrade gracefully for tests
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        """Stub that mirrors :func:`dotenv.load_dotenv` when dependency is absent."""

        return False


def _missing_playwright_check() -> bool:
    """Fallback when ``dynamic_harvest`` cannot be imported locally."""

    print(
        "Playwright verification skipped: neurosales.dynamic_harvest unavailable"
    )
    return False


def _load_ensure_playwright() -> typing.Callable[[], bool]:
    """Return a best-effort Playwright verification function."""

    candidates = (
        "neurosales.dynamic_harvest",
        "neurosales.neurosales.dynamic_harvest",
        "dynamic_harvest",
    )

    for module_name in candidates:
        try:  # pragma: no cover - import paths differ between layouts
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        except Exception:  # pragma: no cover - defensive best effort
            continue

        ensure = getattr(module, "ensure_playwright_browsers", None)
        if callable(ensure):
            return typing.cast(typing.Callable[[], bool], ensure)

    return _missing_playwright_check


ensure_playwright_browsers = _load_ensure_playwright()

load_dotenv()


@dataclass
class SetupHeavyDepsResult:
    """Structured status from :func:`run`."""

    packages_installed: bool
    packages_skipped_reason: typing.Optional[str]
    playwright_install_attempted: bool
    playwright_verified: bool
    embeddings_prefetched: bool
    embeddings_error: typing.Optional[str]


def _log(message: str, logger: typing.Optional[typing.Any]) -> None:
    if logger is None:
        print(message)
    else:
        logger.info(message)


def run(download_only: bool = False, logger=None) -> SetupHeavyDepsResult:
    """Install heavy dependencies and optionally prefetch embeddings."""

    packages_installed = False
    packages_skipped_reason: typing.Optional[str] = None
    playwright_install_attempted = False
    playwright_verified = False
    embeddings_prefetched = False
    embeddings_error: typing.Optional[str] = None

    if not download_only:
        root = Path(__file__).resolve().parents[1]
        reqs = root / "requirements-extra.txt"
        if reqs.is_file():
            _log(f"Installing packages from {reqs}...", logger)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(reqs)]
            )
            packages_installed = True
        else:
            packages_skipped_reason = (
                f"Requirements file {reqs} not found; skipping package installation"
            )
            _log(packages_skipped_reason, logger)

        if shutil.which("playwright"):
            playwright_install_attempted = True
            _log("Installing Playwright browsers...", logger)
            subprocess.check_call(["playwright", "install", "--with-deps"])
            playwright_verified = bool(ensure_playwright_browsers())
            if playwright_verified:
                _log("Playwright installation verified", logger)
            else:
                warning = (
                    "WARNING: Playwright browsers missing. Run 'playwright install --with-deps'"
                )
                if logger is None:
                    print(warning)
                else:
                    logger.warning(warning)
        else:
            _log("Playwright not found; skipping browser install", logger)
    else:
        packages_skipped_reason = "Download-only mode requested"
        _log(packages_skipped_reason, logger)

    try:
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import login
        import os

        login(token=os.getenv("HUGGINGFACE_API_TOKEN"))
        _log("Prefetching embedding model weights...", logger)
        SentenceTransformer("all-MiniLM-L6-v2")
        embeddings_prefetched = True
    except Exception as exc:  # pragma: no cover - optional dependency
        embeddings_error = f"Could not load SentenceTransformer: {exc}"
        if logger is None:
            print(embeddings_error)
        else:
            logger.warning(embeddings_error)

    _log("Heavy dependencies are ready", logger)

    return SetupHeavyDepsResult(
        packages_installed=packages_installed,
        packages_skipped_reason=packages_skipped_reason,
        playwright_install_attempted=playwright_install_attempted,
        playwright_verified=playwright_verified,
        embeddings_prefetched=embeddings_prefetched,
        embeddings_error=embeddings_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup heavy dependencies")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download embedding weights without installing packages",
    )
    args = parser.parse_args()

    run(download_only=args.download_only)


if __name__ == "__main__":
    main()
