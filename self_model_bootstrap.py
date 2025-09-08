from __future__ import annotations

"""Register the Menace system itself as a business model.

This script relies on existing deployment and error tracking bots to populate
Menace's recursive databases. It creates a ``menace`` model entry and then
iterates over all ``*_bot.py`` modules to store bot records, code templates and
associated workflows. Any runtime issues encountered during the process are
logged via :class:`ErrorBot` and linked to the created model.
"""

from pathlib import Path
from dynamic_path_router import resolve_dir

from menace.deployment_bot import DeploymentBot
from menace.error_bot import ErrorBot
from menace.data_bot import DataBot, MetricsDB
from menace.capital_management_bot import CapitalManagementBot
from menace.database_manager import add_model, update_model, DB_PATH
from types import SimpleNamespace


def bootstrap(context_builder: "ContextBuilder" | None = None) -> int:
    """Create the menace model and populate related tables.

    Returns the model identifier from ``models.db``.
    """
    # Insert or fetch the menace model using the standard helper.
    model_id = add_model("menace", source="self", tags="menace", db_path=DB_PATH)

    deployer = DeploymentBot()
    capital_bot = CapitalManagementBot()
    data_bot = DataBot(MetricsDB(), capital_bot=capital_bot)
    cb = context_builder
    if cb is None:
        try:
            from vector_service.context_builder import ContextBuilder

            cb = ContextBuilder(
                bots_db="bots.db",
                code_db="code.db",
                errors_db="errors.db",
                workflows_db="workflows.db",
            )
        except Exception:  # pragma: no cover - best effort fallback
            cb = SimpleNamespace(refresh_db_weights=lambda: None)
    err_bot = ErrorBot(data_bot=data_bot, context_builder=cb)

    bot_files = sorted(resolve_dir(".").glob("*_bot.py"))
    bot_names = [p.stem for p in bot_files]

    # Record a basic workflow covering all bots.
    try:
        wf_ids = deployer._record_workflows(bot_names, model_id, bot_names, [])
    except Exception as exc:  # pragma: no cover - best effort
        err_bot.record_runtime_error(str(exc), model_id=model_id)
        wf_ids = []

    try:
        bot_map = deployer._update_bot_records(
            bot_names,
            model_id=model_id,
            workflows=wf_ids,
            enhancements=[],
            resources={name: {} for name in bot_names},
            levels=None,
            errors=[],
        )
    except Exception as exc:  # pragma: no cover - best effort
        err_bot.record_runtime_error(str(exc), model_id=model_id)
        bot_map = {}

    try:
        deployer._record_code_templates(bot_map, enhancements=[], errors=[])
    except Exception as exc:  # pragma: no cover - best effort
        err_bot.record_runtime_error(
            str(exc), model_id=model_id, bot_ids=list(bot_map.values())
        )

    # Collect baseline metrics for all bots and update ROI
    for name in bot_names:
        try:
            data_bot.collect(name, response_time=0.0, errors=0)
        except Exception as exc:  # pragma: no cover - best effort
            err_bot.record_runtime_error(
                str(exc), model_id=model_id, bot_ids=[bot_map.get(name, "")]
            )

    try:
        capital_bot.update_rois()
        df = data_bot.db.fetch(None)
        if not df.empty:
            roi = float(df["revenue"].sum() - df["expense"].sum())
            update_model(model_id, current_roi=roi, db_path=DB_PATH)
    except Exception as exc:  # pragma: no cover - best effort
        err_bot.record_runtime_error(str(exc), model_id=model_id)

    try:
        err_bot.monitor()
    except Exception as exc:  # pragma: no cover - best effort
        err_bot.record_runtime_error(str(exc), model_id=model_id)

    return model_id


