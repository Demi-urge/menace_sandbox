from __future__ import annotations

"""Simple Flask dashboard with charts for metrics, evolution history and errors."""

import threading
import json
import queue
from typing import Iterable
from pathlib import Path
from types import SimpleNamespace

import logging
from flask import Flask, jsonify, render_template_string, request

from .data_bot import MetricsDB
from .evolution_history_db import EvolutionHistoryDB
from .error_bot import ErrorDB
from .report_generation_bot import ReportGenerationBot, ReportOptions
from .lineage_tracker import LineageTracker
from .scope_utils import Scope, build_scope_clause, apply_scope

try:  # optional dependency
    from .unified_event_bus import UnifiedEventBus  # type: ignore
except Exception:  # pragma: no cover - bus optional
    UnifiedEventBus = None  # type: ignore


_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<h1>Menace Monitoring Dashboard</h1>
<canvas id="metrics" width="400" height="200"></canvas>
<canvas id="evolution" width="400" height="200"></canvas>
<canvas id="errors" width="400" height="200"></canvas>
<div id="lineage"></div>
<script>
function render(nodes){
  const ul=document.createElement('ul');
  nodes.forEach(n=>{
    const li=document.createElement('li');
    const avg =
      (n.average_roi === undefined || n.average_roi === null) ? 'n/a' : n.average_roi;
    const delta =
      (n.roi_delta === undefined || n.roi_delta === null) ? 'n/a' : n.roi_delta;
    const desc = n.mutation_description ? ` - ${n.mutation_description}` : '';
    li.textContent =
      `${n.action} (id=${n.rowid}, avgROI=${avg}, ΔROI=${delta})${desc}`;
    if(n.children && n.children.length){
      li.appendChild(render(n.children));
    }
    ul.appendChild(li);
  });
  return ul;
}
function renderAll(trees){
  const div=document.getElementById('lineage');
  div.innerHTML='';
  Object.entries(trees).forEach(([wid,nodes])=>{
    const h=document.createElement('h3');
    h.textContent=`Workflow ${wid}`;
    div.appendChild(h);
    div.appendChild(render(nodes));
  });
}
let trees={};
async function load() {
  const m = await fetch('/metrics_data').then(r=>r.json());
  const e = await fetch('/evolution_data').then(r=>r.json());
  const err = await fetch('/error_data').then(r=>r.json());
  trees = await fetch('/lineage_data').then(r=>r.json());
  new Chart(
    document.getElementById('metrics'),
    {
      type: 'line',
      data: {
        labels: m.labels,
        datasets: [
          {label: 'CPU', data: m.cpu},
          {label: 'Errors', data: m.errors}
        ]
      }
    }
  );
  new Chart(
    document.getElementById('evolution'),
    {
      type: 'line',
      data: {
        labels: e.labels,
        datasets: [{label: 'ROI', data: e.roi}]
      }
    }
  );
  new Chart(
    document.getElementById('errors'),
    {
      type: 'bar',
      data: {
        labels: err.labels,
        datasets: [{label: 'Count', data: err.count}]
      }
    }
  );
  renderAll(trees);
  const source = new EventSource('/lineage_stream');
  source.onmessage = e => {
    const data = JSON.parse(e.data);
    trees[data.workflow_id]=data.tree;
    renderAll(trees);
  };
}
load();
</script>
</body>
</html>
"""


def MonitoringDashboard(
    metrics_db: MetricsDB | None = None,
    evolution_db: EvolutionHistoryDB | None = None,
    error_db: ErrorDB | None = None,
    orchestrator: object | None = None,
    history_file: str | Path = "roi_history.json",
    event_bus: UnifiedEventBus | None = None,
):
    """Factory returning a metrics dashboard extended with lineage streaming."""

    dash = SimpleNamespace()
    dash.logger = logging.getLogger("MonitoringDashboard")
    dash.app = Flask(__name__)
    dash.history_file = Path(history_file)
    dash.metrics_db = metrics_db or MetricsDB()
    dash.evolution_db = evolution_db or EvolutionHistoryDB()
    dash.error_db = error_db or ErrorDB()
    dash.lineage_tracker = LineageTracker(dash.evolution_db)
    dash.reporter = ReportGenerationBot(dash.metrics_db)
    dash.orchestrator = orchestrator
    dash._report_timer = None  # type: ignore[assignment]
    dash.event_bus = event_bus
    dash._lineage_updates: queue.Queue[list[dict]] = queue.Queue()
    dash._lineage_trees: dict[int, list[dict]] = {}
    dash._lineage_lock = threading.Lock()

    def index() -> tuple[str, int]:
        return render_template_string(_TEMPLATE), 200

    def metrics_data() -> tuple[str, int]:
        rows = dash.metrics_db.fetch(50)
        if hasattr(rows, 'to_dict'):
            data = rows.to_dict('records')  # type: ignore[no-any-return]
        else:
            data = rows
        labels = [r['ts'] for r in data][::-1]
        cpu = [float(r.get('cpu', 0.0)) for r in data][::-1]
        errors = [float(r.get('errors', 0.0)) for r in data][::-1]
        return jsonify({'labels': labels, 'cpu': cpu, 'errors': errors}), 200

    def evolution_data() -> tuple[str, int]:
        rows = dash.evolution_db.fetch(50)
        labels = [r[9] for r in rows][::-1]
        roi = [float(r[3]) for r in rows][::-1]
        return jsonify({'labels': labels, 'roi': roi}), 200

    def error_data() -> tuple[str, int]:
        scope = request.args.get('scope', 'local')
        source_menace_id = request.args.get('source_menace_id')
        menace_id = dash.error_db._menace_id(source_menace_id)
        clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
        query = apply_scope(
            "SELECT error_type, COUNT(*) FROM telemetry",
            clause,
        ) + " GROUP BY error_type"
        cur = dash.error_db.conn.execute(query, params)
        rows = cur.fetchall()
        labels = [r[0] if r[0] is not None else '' for r in rows]
        count = [int(r[1]) for r in rows]
        return jsonify({'labels': labels, 'count': count}), 200

    def lineage_data() -> tuple[str, int]:
        workflow_id = request.args.get('workflow_id', type=int)
        if workflow_id is not None:
            with dash._lineage_lock:
                tree = dash._lineage_trees.get(workflow_id)
            if tree is None:
                tree = dash.lineage_tracker.build_tree(workflow_id)
                with dash._lineage_lock:
                    dash._lineage_trees[workflow_id] = tree
            return jsonify({workflow_id: tree}), 200
        with dash._lineage_lock:
            if not dash._lineage_trees:
                cur = dash.evolution_db.conn.execute(
                    "SELECT DISTINCT workflow_id FROM evolution_history "
                    "WHERE workflow_id IS NOT NULL"
                )
                for row in cur.fetchall():
                    wid = int(row[0])
                    dash._lineage_trees[wid] = dash.lineage_tracker.build_tree(wid)
            data = dict(dash._lineage_trees)
        return jsonify(data), 200

    def _handle_mutation(topic: str, payload: object) -> None:
        try:
            if isinstance(payload, dict):
                workflow_id = payload.get('workflow_id')
                if workflow_id is not None:
                    tree = dash.lineage_tracker.build_tree(int(workflow_id))
                    with dash._lineage_lock:
                        dash._lineage_trees[int(workflow_id)] = tree
                    dash._lineage_updates.put(tree)
        except Exception:
            dash.logger.exception('failed handling mutation event')

    def lineage_stream():
        def _gen():
            while True:
                tree = dash._lineage_updates.get()
                workflow_id = tree[0].get("workflow_id") if tree else None
                payload = {"workflow_id": workflow_id, "tree": tree}
                yield f"data: {json.dumps(payload)}\n\n"

        return dash.app.response_class(_gen(), mimetype='text/event-stream')

    def status() -> tuple[str, int]:
        if dash.orchestrator and hasattr(dash.orchestrator, 'status_summary'):
            try:
                data = dash.orchestrator.status_summary()
                return jsonify(data), 200
            except Exception as exc:
                dash.logger.warning('status_summary failed: %s', exc)
        return jsonify({}), 200

    def schedule_reports(
        metrics: Iterable[str] | None = None,
        *,
        recipients: Iterable[str] | None = None,
        interval: int = 60 * 60 * 24,
    ) -> None:
        options = ReportOptions(metrics=list(metrics or ['cpu', 'errors']))
        if recipients:
            options.recipients = list(recipients)
        dash.reporter.schedule(options, interval=interval)

    # bind methods and routes
    dash.index = index
    dash.metrics_data = metrics_data
    dash.evolution_data = evolution_data
    dash.error_data = error_data
    dash.lineage_data = lineage_data
    dash._handle_mutation = _handle_mutation
    dash.lineage_stream = lineage_stream
    dash.status = status
    dash.schedule_reports = schedule_reports

    dash.app.add_url_rule('/', 'index', index)
    dash.app.add_url_rule('/metrics_data', 'metrics_data', metrics_data)
    dash.app.add_url_rule('/evolution_data', 'evolution_data', evolution_data)
    dash.app.add_url_rule('/error_data', 'error_data', error_data)
    dash.app.add_url_rule('/lineage_data', 'lineage_data', lineage_data)
    dash.app.add_url_rule('/status', 'status', status)
    dash.app.add_url_rule('/lineage_stream', 'lineage_stream', lineage_stream)

    if dash.event_bus is not None:
        dash.event_bus.subscribe('mutation_recorded', _handle_mutation)

    return dash


__all__ = ["MonitoringDashboard"]
