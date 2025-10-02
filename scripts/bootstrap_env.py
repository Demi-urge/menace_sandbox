#!/usr/bin/env python3
"""Bootstrap the Menace environment and verify dependencies.

Run ``python scripts/bootstrap_env.py`` to install required tooling and
configuration.  Pass ``--skip-stripe-router`` to bypass the Stripe router
startup verification when working offline or without Stripe credentials.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-stripe-router",
        action="store_true",
        help=(
            "Bypass the Stripe router startup verification. Useful when Stripe "
            "credentials are unavailable during local bootstraps."
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            "Optional path to the environment file that should receive generated "
            "defaults.  When omitted the bootstrap process falls back to the "
            "standard discovery rules in bootstrap_defaults."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    # Explicitly disable safe mode regardless of existing variables
    os.environ["MENACE_SAFE"] = "0"
    # Menace sandbox environments often lack Hugging Face credentials; suppress
    # the warning during bootstraps so local runs remain noise free.
    os.environ.setdefault("MENACE_ALLOW_MISSING_HF_TOKEN", "1")
    # Ensure bootstrap runs without interactive prompts when stdin is a TTY.
    # CI environments as well as our automated tests execute this script in
    # non-interactive shells and would otherwise hang waiting for user input
    # when optional environment variables are missing.  ``startup_checks``
    # honours ``MENACE_NON_INTERACTIVE`` so set it proactively.
    os.environ.setdefault("MENACE_NON_INTERACTIVE", "1")
    if args.skip_stripe_router:
        os.environ["MENACE_SKIP_STRIPE_ROUTER"] = "1"
    if args.env_file:
        os.environ["MENACE_ENV_FILE"] = str(args.env_file.resolve())

    from menace.bootstrap_policy import PolicyLoader
    from menace.environment_bootstrap import EnvironmentBootstrapper
    import startup_checks
    from startup_checks import run_startup_checks
    from menace.bootstrap_defaults import ensure_bootstrap_defaults

    created, env_file = ensure_bootstrap_defaults(
        startup_checks.REQUIRED_VARS,
        repo_root=_REPO_ROOT,
        env_file=args.env_file,
    )
    if created:
        logging.getLogger(__name__).info(
            "Persisted generated defaults to %s", env_file
        )

    loader = PolicyLoader()
    auto_install = startup_checks.auto_install_enabled()
    env_requested = os.getenv("MENACE_BOOTSTRAP_PROFILE")
    requested = env_requested or ("minimal" if not auto_install else None)
    policy = loader.resolve(
        requested=requested,
        auto_install_enabled=auto_install,
    )
    run_startup_checks(skip_stripe_router=args.skip_stripe_router, policy=policy)
    EnvironmentBootstrapper(policy=policy).bootstrap()


if __name__ == "__main__":
    main()
