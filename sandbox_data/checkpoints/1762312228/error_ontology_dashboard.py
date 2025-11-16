from __future__ import annotations

"""Dashboard for exploring error ontology statistics."""

import os
import uuid
import json
from pathlib import Path
from typing import Optional

from flask import jsonify, render_template_string, request

from db_router import init_db_router
from .metrics_dashboard import MetricsDashboard
from .knowledge_graph import KnowledgeGraph
from .scope_utils import Scope, build_scope_clause, apply_scope
from typing import TYPE_CHECKING

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

try:  # pragma: no cover - optional predictor dependency
    from .error_cluster_predictor import ErrorClusterPredictor
except Exception:  # pragma: no cover - fallback for tests
    class ErrorClusterPredictor:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

        def predict_high_risk_modules(
            self, *, min_cluster_size: int = 2, top_n: int = 5
        ) -> list[str]:
            return []

if TYPE_CHECKING:  # pragma: no cover - for type checking
    from .error_bot import ErrorDB

MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv(
    "MENACE_LOCAL_DB_PATH", str(resolve_path(f"menace_{MENACE_ID}_local.db"))
)
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
GLOBAL_ROUTER = init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)


_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<h1>Error Ontology Dashboard</h1>
<canvas id="by_category" width="400" height="200"></canvas>
<canvas id="by_module" width="400" height="200"></canvas>
<canvas id="by_cause" width="400" height="200"></canvas>
<canvas id="success_by_category" width="400" height="200"></canvas>
<canvas id="predicted_modules" width="400" height="200"></canvas>
<script>
async function load(){
  const c = await fetch('/category_data').then(r=>r.json());
  const m = await fetch('/module_data').then(r=>r.json());
  const k = await fetch('/cause_data').then(r=>r.json());
  const s = await fetch('/category_success').then(r=>r.json());
  const p = await fetch('/predicted_modules').then(r=>r.json());
  new Chart(document.getElementById('by_category'), {
    type:'bar',
    data:{labels:c.labels,datasets:[{label:'Count',data:c.count}]}
  });
  new Chart(document.getElementById('by_module'), {
    type:'bar',
    data:{labels:m.labels,datasets:[{label:'Count',data:m.count}]}
  });
  new Chart(document.getElementById('by_cause'), {
    type:'bar',
    data:{labels:k.labels,datasets:[{label:'Count',data:k.count}]}
  });
  new Chart(document.getElementById('success_by_category'), {
    type:'bar',
    data:{labels:s.labels,datasets:[{label:'Success Rate',data:s.rate}]},
    options:{scales:{y:{beginAtZero:true,max:1}}}
  });
  new Chart(document.getElementById('predicted_modules'), {
    type:'bar',
    data:{labels:p.labels,datasets:[{label:'Risk Rank',data:p.rank}]}
  });
}
load();
</script>
</body>
</html>
"""


class ErrorOntologyDashboard(MetricsDashboard):
    """Dashboard exposing error counts by category and module."""

    def __init__(
        self,
        error_db: 'ErrorDB | None' = None,
        graph: Optional[KnowledgeGraph] = None,
        *,
        history_file: str | Path = "roi_history.json",
    ) -> None:
        super().__init__(history_file)
        if error_db is None:
            from .error_bot import ErrorDB as _ErrorDB
            error_db = _ErrorDB()
        self.error_db = error_db
        self.graph = graph or KnowledgeGraph()
        self.predictor = ErrorClusterPredictor(self.graph, self.error_db)
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/category_data', 'category_data', self.category_data)
        self.app.add_url_rule('/module_data', 'module_data', self.module_data)
        self.app.add_url_rule('/cause_data', 'cause_data', self.cause_data)
        self.app.add_url_rule('/category_success', 'category_success', self.category_success)
        self.app.add_url_rule('/predicted_modules', 'predicted_modules', self.predicted_modules)

    # ------------------------------------------------------------------
    def index(self) -> tuple[str, int]:
        return render_template_string(_TEMPLATE), 200

    def category_data(
        self,
        scope: str | None = None,
        source_menace_id: str | None = None,
    ) -> tuple[str, int]:
        scope = scope or request.args.get("scope", "local")
        source_menace_id = source_menace_id or request.args.get("source_menace_id")
        menace_id = self.error_db._menace_id(source_menace_id)
        clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
        query = apply_scope(
            'SELECT COALESCE(category, ""), COUNT(*) FROM telemetry', clause
        )
        query += ' GROUP BY category'
        cur = self.error_db.conn.execute(query, params)
        rows = cur.fetchall()
        labels = [r[0] for r in rows]
        count = [int(r[1]) for r in rows]
        return jsonify({'labels': labels, 'count': count}), 200

    def module_data(
        self,
        scope: str | None = None,
        source_menace_id: str | None = None,
    ) -> tuple[str, int]:
        scope = scope or request.args.get("scope", "local")
        source_menace_id = source_menace_id or request.args.get("source_menace_id")
        menace_id = self.error_db._menace_id(source_menace_id)
        clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
        query = apply_scope(
            'SELECT COALESCE(module, ""), COUNT(*) FROM telemetry', clause
        )
        query += ' GROUP BY module'
        cur = self.error_db.conn.execute(query, params)
        rows = cur.fetchall()
        labels = [r[0] for r in rows]
        count = [int(r[1]) for r in rows]
        return jsonify({'labels': labels, 'count': count}), 200

    def cause_data(
        self,
        scope: str | None = None,
        source_menace_id: str | None = None,
    ) -> tuple[str, int]:
        scope = scope or request.args.get("scope", "local")
        source_menace_id = source_menace_id or request.args.get("source_menace_id")
        menace_id = self.error_db._menace_id(source_menace_id)
        clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
        query = apply_scope(
            'SELECT COALESCE(cause, ""), COUNT(*) FROM telemetry', clause
        )
        query += ' GROUP BY cause'
        cur = self.error_db.conn.execute(query, params)
        rows = cur.fetchall()
        labels = [r[0] for r in rows]
        count = [int(r[1]) for r in rows]
        return jsonify({'labels': labels, 'count': count}), 200

    def category_success(
        self,
        scope: str | None = None,
        source_menace_id: str | None = None,
    ) -> tuple[str, int]:
        scope = scope or request.args.get("scope", "local")
        source_menace_id = source_menace_id or request.args.get("source_menace_id")
        menace_id = self.error_db._menace_id(source_menace_id)
        clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
        query = apply_scope(
            "SELECT COALESCE(category, ''), "
            "AVG(CASE WHEN resolution_status='successful' "
            "THEN 1.0 ELSE 0.0 END) FROM telemetry",
            clause,
        )
        query += ' GROUP BY category'
        cur = self.error_db.conn.execute(query, params)
        rows = cur.fetchall()
        labels = [r[0] for r in rows]
        rate = [float(r[1] or 0.0) for r in rows]
        return jsonify({'labels': labels, 'rate': rate}), 200

    def predicted_modules(self) -> tuple[str, int]:
        modules = self.predictor.predict_high_risk_modules()
        rank = list(range(len(modules), 0, -1))
        return jsonify({'labels': modules, 'rank': rank}), 200

    def generate_report(
        self,
        path: str | Path = 'error_ontology_report.json',
        *,
        scope: str = 'local',
        source_menace_id: str | None = None,
    ) -> Path:
        """Generate a JSON report of error counts."""
        # Keep graph in sync with latest stats
        try:
            self.graph.update_error_stats(self.error_db)
        except Exception:
            pass
        menace_id = self.error_db._menace_id(source_menace_id)
        clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
        query = apply_scope(
            'SELECT COALESCE(category, "") as category, '
            'COALESCE(module, "") as module, '
            'COUNT(*) as count FROM telemetry',
            clause,
        )
        query += ' GROUP BY category, module'
        cur = self.error_db.conn.execute(query, params)
        rows = cur.fetchall()
        data = [
            {"category": r[0], "module": r[1], "count": int(r[2])}
            for r in rows
        ]
        cause_query = apply_scope(
            'SELECT COALESCE(cause, "") as cause, COUNT(*) as count FROM telemetry',
            clause,
        )
        cause_query += ' GROUP BY cause'
        cause_cur = self.error_db.conn.execute(cause_query, params)
        cause_rows = cause_cur.fetchall()
        causes = [
            {"cause": r[0], "count": int(r[1])}
            for r in cause_rows
        ]
        dest = Path(path)
        dest.write_text(
            json.dumps({"error_stats": data, "cause_stats": causes}, indent=2)
        )
        return dest


__all__ = ["ErrorOntologyDashboard", "cli", "main"]


def cli(argv: list[str] | None = None) -> None:
    """Generate an error ontology report from the database."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='error_ontology_report.json')
    parser.add_argument('--scope', default='local', choices=[s.value for s in Scope])
    parser.add_argument('--source-menace-id')
    args = parser.parse_args(argv)

    dash = ErrorOntologyDashboard()
    dash.generate_report(
        args.output,
        scope=args.scope,
        source_menace_id=args.source_menace_id,
    )


def main(argv: list[str] | None = None) -> None:
    cli(argv)


if __name__ == '__main__':  # pragma: no cover - entry point
    main()
