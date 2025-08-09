from __future__ import annotations

"""Dashboard for exploring error ontology statistics."""

from pathlib import Path
from typing import Optional
import json

from flask import jsonify, render_template_string

from .metrics_dashboard import MetricsDashboard
from .error_bot import ErrorDB
from .knowledge_graph import KnowledgeGraph


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
<script>
async function load(){
  const c = await fetch('/category_data').then(r=>r.json());
  const m = await fetch('/module_data').then(r=>r.json());
  new Chart(document.getElementById('by_category'), {type:'bar',data:{labels:c.labels,datasets:[{label:'Count',data:c.count}]}});
  new Chart(document.getElementById('by_module'), {type:'bar',data:{labels:m.labels,datasets:[{label:'Count',data:m.count}]}});
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
        error_db: Optional[ErrorDB] = None,
        graph: Optional[KnowledgeGraph] = None,
        *,
        history_file: str | Path = "roi_history.json",
    ) -> None:
        super().__init__(history_file)
        self.error_db = error_db or ErrorDB()
        self.graph = graph or KnowledgeGraph()
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/category_data', 'category_data', self.category_data)
        self.app.add_url_rule('/module_data', 'module_data', self.module_data)

    # ------------------------------------------------------------------
    def index(self) -> tuple[str, int]:
        return render_template_string(_TEMPLATE), 200

    def category_data(self) -> tuple[str, int]:
        cur = self.error_db.conn.execute(
            'SELECT COALESCE(category, ""), COUNT(*) FROM telemetry GROUP BY category'
        )
        rows = cur.fetchall()
        labels = [r[0] for r in rows]
        count = [int(r[1]) for r in rows]
        return jsonify({'labels': labels, 'count': count}), 200

    def module_data(self) -> tuple[str, int]:
        cur = self.error_db.conn.execute(
            'SELECT COALESCE(module, ""), COUNT(*) FROM telemetry GROUP BY module'
        )
        rows = cur.fetchall()
        labels = [r[0] for r in rows]
        count = [int(r[1]) for r in rows]
        return jsonify({'labels': labels, 'count': count}), 200

    def generate_report(self, path: str | Path = 'error_ontology_report.json') -> Path:
        """Generate a JSON report of error counts."""
        # Keep graph in sync with latest stats
        try:
            self.graph.update_error_stats(self.error_db)
        except Exception:
            pass
        cur = self.error_db.conn.execute(
            'SELECT COALESCE(category, "") as category, COALESCE(module, "") as module, COUNT(*) as count\n'
            'FROM telemetry GROUP BY category, module'
        )
        rows = cur.fetchall()
        data = [
            {"category": r[0], "module": r[1], "count": int(r[2])}
            for r in rows
        ]
        dest = Path(path)
        dest.write_text(json.dumps({"error_stats": data}, indent=2))
        return dest


__all__ = ["ErrorOntologyDashboard", "cli", "main"]


def cli(argv: list[str] | None = None) -> None:
    """Generate an error ontology report from the database."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='error_ontology_report.json')
    args = parser.parse_args(argv)

    dash = ErrorOntologyDashboard()
    dash.generate_report(args.output)


def main(argv: list[str] | None = None) -> None:
    cli(argv)


if __name__ == '__main__':  # pragma: no cover - entry point
    main()
