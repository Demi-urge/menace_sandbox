from __future__ import annotations

"""Helpers for cross database queries across Menace subsystems."""

from typing import Any, Dict, Iterable, List, Optional

import sqlite3
import logging

from .bot_registry import BotRegistry
from .neuroplasticity import PathwayDB
from .databases import MenaceDB
from .research_aggregator_bot import InfoDB
from gpt_memory_interface import GPTMemoryInterface
from .scope_utils import Scope


logger = logging.getLogger(__name__)
def related_workflows(
    bot_name: str,
    *,
    registry: BotRegistry,
    menace_db: MenaceDB,
    pathway_db: PathwayDB | None = None,
    depth: int = 1,
) -> List[str]:
    """Return workflow names connected to ``bot_name``.

    The search expands across ``BotRegistry`` edges up to ``depth`` and ranks
    results using myelination scores from ``PathwayDB`` when available.
    """

    bot_scores: Dict[str, float] = {bot_name: 1.0}
    for name, _ in registry.connections(bot_name, depth):
        bot_scores[name] = bot_scores.get(name, 0.0) + 1.0
    bot_names = set(bot_scores)
    bot_ids: set[int] = set()

    scores: Dict[str, float] = {}
    with menace_db.engine.connect() as conn:
        for name in bot_names:
            row = (
                conn.execute(
                    menace_db.bots.select().where(
                        menace_db.bots.c.bot_name == name
                    )
                )
                .mappings()
                .fetchone()
            )
            if not row:
                continue
            bot_id = int(row["bot_id"])
            bot_ids.add(bot_id)
            bot_ids.add(bot_id)
            wf_rows = conn.execute(
                menace_db.workflow_bots.select().where(
                    menace_db.workflow_bots.c.bot_id == bot_id
                )
            ).fetchall()
            wf_ids = [int(r[0]) for r in wf_rows]
            if not wf_ids:
                continue
            w_rows = (
                conn.execute(
                    menace_db.workflows.select().where(
                        menace_db.workflows.c.workflow_id.in_(wf_ids)
                    )
                )
                .mappings()
                .fetchall()
            )
            for wf in w_rows:
                wname = wf["workflow_name"]
                score = 0.0
                if pathway_db:
                    sim = pathway_db.similar_actions(wname, limit=1)
                    if sim:
                        score = float(sim[0][1])
                if scores.get(wname, 0.0) < score:
                    scores[wname] = score
                else:
                    scores.setdefault(wname, score)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ordered]


def similar_code_snippets(
    template: str,
    *,
    menace_db: MenaceDB,
    registry: BotRegistry | None = None,
    pathway_db: PathwayDB | None = None,
    limit: int = 5,
) -> List[Dict[str, object]]:
    """Return code snippets matching ``template``.

    ``MenaceDB`` tables are queried for summaries or template types containing
    ``template``. Associated bots are expanded using ``BotRegistry`` to include
    their snippets as well. ``PathwayDB`` similarity helps rank the results.
    """

    template_l = f"%{template}%"
    code_ids: set[int] = set()
    records: Dict[int, Dict[str, object]] = {}

    with menace_db.engine.connect() as conn:
        rows = (
            conn.execute(
                menace_db.code.select().where(
                    (menace_db.code.c.code_summary.like(template_l))
                    | (menace_db.code.c.template_type.like(template_l))
                )
            )
            .mappings()
            .fetchall()
        )
        for row in rows:
            cid = int(row["code_id"])
            records[cid] = dict(row)
            code_ids.add(cid)
            bot_rows = conn.execute(
                menace_db.code_bots.select().where(
                    menace_db.code_bots.c.code_id == cid
                )
            ).fetchall()
            bot_ids = [int(r[1]) for r in bot_rows]
            if registry:
                for bid in bot_ids:
                    b_row = (
                        conn.execute(
                            menace_db.bots.select().where(
                                menace_db.bots.c.bot_id == bid
                            )
                        )
                        .mappings()
                        .fetchone()
                    )
                    if not b_row:
                        continue
                    bname = b_row["bot_name"]
                    for nb, _ in registry.connections(bname):
                        nb_row = (
                            conn.execute(
                                menace_db.bots.select().where(
                                    menace_db.bots.c.bot_name == nb
                                )
                            )
                            .mappings()
                            .fetchone()
                        )
                        if not nb_row:
                            continue
                        nbid = int(nb_row["bot_id"])
                        extra = conn.execute(
                            menace_db.code_bots.select().where(
                                menace_db.code_bots.c.bot_id == nbid
                            )
                        ).fetchall()
                        code_ids.update(int(e[0]) for e in extra)

        for cid in list(code_ids):
            if cid in records:
                continue
            row = (
                conn.execute(
                    menace_db.code.select().where(
                        menace_db.code.c.code_id == cid
                    )
                )
                .mappings()
                .fetchone()
            )
            if row:
                records[cid] = dict(row)

    results = []
    for cid, rec in records.items():
        score = 0.0
        if pathway_db:
            sim = pathway_db.similar_actions(str(rec.get("code_summary", "")), limit=1)
            if sim:
                score = float(sim[0][1])
        rec["score"] = score
        results.append(rec)

    results.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return results[:limit]


def related_resources(
    bot_name: str,
    *,
    registry: BotRegistry,
    menace_db: MenaceDB,
    info_db: Optional[InfoDB] = None,
    memory_mgr: Optional[GPTMemoryInterface] = None,
    pathway_db: PathwayDB | None = None,
    depth: int = 1,
) -> Dict[str, List[str]]:
    """Return resources connected to ``bot_name`` across databases."""

    bot_scores: Dict[str, float] = {bot_name: 1.0}
    for name, weight in registry.connections(bot_name, depth):
        bot_scores[name] = bot_scores.get(name, 0.0) + weight
    bot_names = set(bot_scores)

    workflows: Dict[str, float] = {}
    infos: Dict[str, float] = {}
    memory_keys: Dict[str, float] = {}
    bot_ids: set[int] = set()
    info_ids: set[int] = set()

    with menace_db.engine.connect() as conn:
        for name in bot_names:
            row = (
                conn.execute(
                    menace_db.bots.select().where(
                        menace_db.bots.c.bot_name == name
                    )
                )
                .mappings()
                .fetchone()
            )
            if not row:
                continue
            bot_id = int(row["bot_id"])

            wf_rows = conn.execute(
                menace_db.workflow_bots.select().where(
                    menace_db.workflow_bots.c.bot_id == bot_id
                )
            ).fetchall()
            w_ids = [int(r[0]) for r in wf_rows]
            if w_ids:
                w_rows = (
                    conn.execute(
                        menace_db.workflows.select().where(
                            menace_db.workflows.c.workflow_id.in_(w_ids)
                        )
                    )
                    .mappings()
                    .fetchall()
                )
                for w in w_rows:
                    workflows[w["workflow_name"]] = workflows.get(w["workflow_name"], 0.0) + 1.0

            if hasattr(menace_db, "information_bots"):
                i_rows = conn.execute(
                    menace_db.information_bots.select().where(
                        menace_db.information_bots.c.bot_id == bot_id
                    )
                ).fetchall()
                ids = [int(r[0]) for r in i_rows]
                if ids:
                    info_ids.update(ids)
                    inf_rows = (
                        conn.execute(
                            menace_db.information.select().where(
                                menace_db.information.c.info_id.in_(ids)
                            )
                        )
                        .mappings()
                        .fetchall()
                    )
                    for i in inf_rows:
                        key = i["summary"] or str(i["info_id"])
                        infos[key] = infos.get(key, 0.0) + 1.0

    if info_db:
        for name in bot_names:
            try:
                items = info_db.search(name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("info_db search failed for %s: %s", name, exc)
                continue
            for it in items:
                key = it.title or it.topic
                infos[key] = infos.get(key, 0.0) + 1.0
                if getattr(it, "item_id", 0):
                    info_ids.add(it.item_id)

    if memory_mgr:
        for name in bot_names:
            try:
                entries = memory_mgr.search_context(name, tags=[name])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "memory search_context failed for %s: %s", name, exc
                )
                continue
            for e in entries:
                key = getattr(e, "key", getattr(e, "prompt", ""))
                if key:
                    memory_keys[key] = memory_keys.get(key, 0.0) + 1.0

        # Legacy fallback when underlying manager exposes raw connection
        if hasattr(memory_mgr, "conn"):
            cur = memory_mgr.conn.cursor()
            if bot_ids:
                marks = ",".join("?" for _ in bot_ids)
                try:
                    rows = cur.execute(
                        f"SELECT key FROM memory WHERE bot_id IN ({marks})",
                        tuple(bot_ids),
                    ).fetchall()
                    for r in rows:
                        memory_keys[r[0]] = memory_keys.get(r[0], 0.0) + 1.0
                except Exception as exc:
                    logger.exception(
                        "failed to fetch memory keys for bot ids: %s", exc
                    )
            if info_ids:
                marks = ",".join("?" for _ in info_ids)
                try:
                    rows = cur.execute(
                        f"SELECT key FROM memory WHERE info_id IN ({marks})",
                        tuple(info_ids),
                    ).fetchall()
                    for r in rows:
                        memory_keys[r[0]] = memory_keys.get(r[0], 0.0) + 1.0
                except Exception as exc:
                    logger.exception(
                        "failed to fetch memory keys for info ids: %s", exc
                    )

    if pathway_db:
        for wf in list(workflows):
            sim = pathway_db.similar_actions(wf, limit=1)
            workflows[wf] += float(sim[0][1]) if sim else 0.0

    bots_sorted = [b for b, _ in sorted(bot_scores.items(), key=lambda x: x[1], reverse=True)]
    wfs_sorted = [w for w, _ in sorted(workflows.items(), key=lambda x: x[1], reverse=True)]
    infos_sorted = [i for i, _ in sorted(infos.items(), key=lambda x: x[1], reverse=True)]
    mem_sorted = [m for m, _ in sorted(memory_keys.items(), key=lambda x: x[1], reverse=True)]

    return {
        "bots": bots_sorted,
        "workflows": wfs_sorted,
        "information": infos_sorted,
        "memory": mem_sorted,
    }


def entry_workflow_features(
    entry: dict,
    *,
    registry: BotRegistry,
    menace_db: MenaceDB,
    pathway_db: PathwayDB | None = None,
    info_db: Optional[InfoDB] = None,
    memory_mgr: Optional[GPTMemoryInterface] = None,
    depth: int = 1,
) -> List[str]:
    """Return workflow names linked to a new database ``entry``.

    The helper extracts bot identifiers from ``entry`` and reuses
    :func:`related_workflows` to locate connected workflows.
    When ``summary`` text is available it also expands via
    :func:`similar_code_snippets` to include bots associated with
    matching code records.
    """

    bots: set[str] = set()

    if "bot" in entry:
        bots.add(str(entry["bot"]))
    if "bot_name" in entry:
        bots.add(str(entry["bot_name"]))
    if "bot_id" in entry:
        try:
            bid = int(entry["bot_id"])
            with menace_db.engine.connect() as conn:
                row = (
                    conn.execute(
                        menace_db.bots.select().where(
                            menace_db.bots.c.bot_id == bid
                        )
                    )
                    .mappings()
                    .fetchone()
                )
                if row:
                    bots.add(row["bot_name"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to resolve bot_id %s: %s", entry.get("bot_id"), exc)
    tags = entry.get("tags")
    if isinstance(tags, str):
        bots.update(t.strip() for t in tags.split(",") if t.strip())
    elif isinstance(tags, (list, tuple)):
        bots.update(str(t) for t in tags)

    if info_db and entry.get("info_id"):
        try:
            wids = info_db.workflows_for(int(entry["info_id"]))
            if wids:
                with menace_db.engine.connect() as conn:
                    rows = (
                        conn.execute(
                            menace_db.workflows.select().where(
                                menace_db.workflows.c.workflow_id.in_(wids)
                            )
                        )
                        .mappings()
                        .fetchall()
                    )
                return [r["workflow_name"] for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed workflow lookup for info_id %s: %s", entry.get("info_id"), exc)

    scores: Dict[str, float] = {}

    for bot in bots:
        try:
            wfs = related_workflows(
                bot,
                registry=registry,
                menace_db=menace_db,
                pathway_db=pathway_db,
                depth=depth,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("related_workflows failed for %s: %s", bot, exc)
            continue
        for wf in wfs:
            scores[wf] = scores.get(wf, 0.0) + 1.0

    if entry.get("summary"):
        try:
            snippets = similar_code_snippets(
                str(entry["summary"]),
                menace_db=menace_db,
                registry=registry,
                pathway_db=pathway_db,
            )
            bot_ids = {
                int(s.get("bot_id", 0))
                for s in snippets
                if s.get("bot_id") is not None
            }
            if bot_ids:
                with menace_db.engine.connect() as conn:
                    rows = conn.execute(
                        menace_db.workflow_bots.select().where(
                            menace_db.workflow_bots.c.bot_id.in_(list(bot_ids))
                        )
                    ).fetchall()
                    wf_ids = [int(r[0]) for r in rows]
                    if wf_ids:
                        w_rows = (
                            conn.execute(
                                menace_db.workflows.select().where(
                                    menace_db.workflows.c.workflow_id.in_(wf_ids)
                                )
                            )
                            .mappings()
                            .fetchall()
                        )
                        for wf in w_rows:
                            name = wf["workflow_name"]
                            scores[name] = scores.get(name, 0.0) + 1.0
        except Exception as exc:  # noqa: BLE001
            logger.warning("similar_code_snippets failed: %s", exc)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ordered]


def workflow_roi_stats(
    workflow_name: str,
    roi_db: ROIDB,
    metrics_db: MetricsDB,
    *,
    limit: int | None = 50,
    source_menace_id: Any | None = None,
    scope: Scope | str = Scope.LOCAL,
) -> Dict[str, float]:
    """Return aggregated ROI, CPU seconds and API cost for *workflow_name*.

    The helper combines ROI history recorded by :class:`ROIDB` with any
    evaluation metrics stored in :class:`MetricsDB`. When available, results are
    summed over the latest ``limit`` entries.
    """

    revenue = api_cost = cpu_seconds = 0.0
    try:
        df = roi_db.history(workflow_name, limit=limit or 50)
        if not getattr(df, "empty", False):
            revenue = float(df["revenue"].sum())
            api_cost = float(df["api_cost"].sum())
            cpu_seconds = float(df.get("cpu_seconds", df.get("duration", 0)).sum())
    except Exception as exc:  # noqa: BLE001
        logger.warning("ROI history lookup failed for %s: %s", workflow_name, exc)

    try:
        rows = metrics_db.fetch_eval(
            workflow_name, source_menace_id=source_menace_id, scope=scope
        )
        for _, metric, value, _ in rows:
            if metric == "duration":
                cpu_seconds += float(value)
            elif metric in {"api_cost", "expense"}:
                api_cost += float(value)
    except Exception as exc:  # noqa: BLE001
        logger.warning("metrics fetch failed for %s: %s", workflow_name, exc)

    roi = revenue - api_cost
    return {"roi": roi, "cpu_seconds": cpu_seconds, "api_cost": api_cost}


def rank_workflows(
    workflows: Iterable[str],
    roi_db: ROIDB,
    metrics_db: MetricsDB,
    *,
    source_menace_id: Any | None = None,
    scope: Scope | str = Scope.LOCAL,
) -> List[tuple[str, float]]:
    """Return workflows ranked by ROI per CPU second."""

    scores = {}
    for wf in workflows:
        stats = workflow_roi_stats(
            wf,
            roi_db,
            metrics_db,
            source_menace_id=source_menace_id,
            scope=scope,
        )
        cpu = stats["cpu_seconds"] or 1.0
        scores[wf] = stats["roi"] / cpu
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ordered


def bot_roi_stats(
    bot_name: str,
    roi_db: ROIDB,
    metrics_db: MetricsDB,
    *,
    limit: int | None = 50,
    source_menace_id: Any | None = None,
    scope: Scope | str = Scope.LOCAL,
) -> Dict[str, float]:
    """Return aggregated ROI, CPU seconds and API cost for ``bot_name``."""

    revenue = api_cost = cpu_seconds = 0.0
    try:
        df = roi_db.history(bot_name, limit=limit or 50)
        if not getattr(df, "empty", False):
            revenue = float(df["revenue"].sum())
            api_cost = float(df["api_cost"].sum())
            cpu_seconds = float(df.get("cpu_seconds", df.get("duration", 0)).sum())
    except Exception as exc:  # noqa: BLE001
        logger.warning("ROI history lookup failed for bot %s: %s", bot_name, exc)

    try:
        rows = metrics_db.fetch_eval(
            bot_name, source_menace_id=source_menace_id, scope=scope
        )
        for _, metric, value, _ in rows:
            if metric == "duration":
                cpu_seconds += float(value)
            elif metric in {"api_cost", "expense"}:
                api_cost += float(value)
    except Exception as exc:  # noqa: BLE001
        logger.warning("metrics fetch failed for bot %s: %s", bot_name, exc)

    roi = revenue - api_cost
    return {"roi": roi, "cpu_seconds": cpu_seconds, "api_cost": api_cost}


def rank_bots(
    bots: Iterable[str],
    roi_db: ROIDB,
    metrics_db: MetricsDB,
    *,
    source_menace_id: Any | None = None,
    scope: Scope | str = Scope.LOCAL,
) -> List[tuple[str, float]]:
    """Return bots ranked by ROI per CPU second."""

    scores = {}
    for bot in bots:
        stats = bot_roi_stats(
            bot,
            roi_db,
            metrics_db,
            source_menace_id=source_menace_id,
            scope=scope,
        )
        cpu = stats["cpu_seconds"] or 1.0
        scores[bot] = stats["roi"] / cpu
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ordered


__all__ = [
    "related_workflows",
    "similar_code_snippets",
    "related_resources",
    "entry_workflow_features",
    "workflow_roi_stats",
    "rank_workflows",
    "bot_roi_stats",
    "rank_bots",
]
