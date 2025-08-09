from __future__ import annotations

"""Predict future error probabilities from telemetry metrics."""

from typing import List, Tuple, TYPE_CHECKING

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore

import numpy as np
import hashlib

from .data_bot import MetricsDB
from .knowledge_graph import KnowledgeGraph

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .error_bot import ErrorDB


class ErrorForecaster:
    """LSTM-based sequence model for error forecasting."""

    def __init__(
        self,
        metrics_db: MetricsDB,
        *,
        seq_len: int = 5,
        hidden: int = 32,
        epochs: int = 5,
        extra_features: bool = True,
        dropout: float = 0.1,
        model: str = "lstm",
        error_db: "ErrorDB | None" = None,
        include_clusters: bool = False,
        graph: KnowledgeGraph | None = None,
    ) -> None:
        self.metrics_db = metrics_db
        self.seq_len = seq_len
        self.hidden = hidden
        self.epochs = epochs
        self.extra_features = extra_features
        self.error_db = error_db
        self.include_clusters = include_clusters and error_db is not None
        self.graph = graph
        self.include_graph = graph is not None
        self.use_torch = torch is not None and nn is not None
        self.model_type = model
        self.default_rate = 0.0
        self.transformer = None
        self.lstm = None
        if self.use_torch:
            # features per timestep: errors, cpu, memory, ROI and extra signals
            input_size = (
                7
                + (1 if self.include_clusters else 0)
                + (2 if extra_features else 0)
                + (2 if self.include_graph else 0)
            )
            self.dropout = nn.Dropout(dropout)
            if model == "transformer":
                layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1, dropout=dropout)
                self.transformer = nn.TransformerEncoder(layer, num_layers=2)
                self.fc = nn.Linear(input_size, 1)
                params = list(self.transformer.parameters()) + list(self.fc.parameters())
            else:
                self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
                self.fc = nn.Linear(hidden, 1)
                params = list(self.lstm.parameters()) + list(self.fc.parameters())
            self.optim = torch.optim.Adam(params, lr=0.01)
            self.loss = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    def _dataset(self) -> List[Tuple[str, List[List[float]], float]]:
        df = self.metrics_db.fetch(None)
        data: List[Tuple[str, List[List[float]], float]] = []
        if hasattr(df, "empty") and not getattr(df, "empty", True):
            df = df.sort_values("ts")
            for bot, group in df.groupby("bot"):
                errs = group["errors"].tolist()
                cpu = group["cpu"].tolist()
                mem = group["memory"].tolist()
                disk = group["disk_io"].tolist() if "disk_io" in group else [0.0] * len(cpu)
                net = group["net_io"].tolist() if "net_io" in group else [0.0] * len(cpu)
                roi = (group["revenue"] - group["expense"]).tolist()
                wf_val = float(len(str(bot)))
                bot_id = float(abs(hash(str(bot))) % 1000) / 1000.0
                cluster_feat = self._cluster_feature(str(bot))
                graph_feat = self._graph_features(str(bot))
                for i in range(len(errs) - self.seq_len):
                    seq = []
                    for j in range(i, i + self.seq_len):
                        row = [
                            float(errs[j]),
                            float(cpu[j]),
                            float(mem[j]),
                            float(roi[j]),
                            wf_val,
                            float(disk[j] + net[j] - (disk[j - 1] + net[j - 1])) if j > 0 else 0.0,
                            bot_id,
                        ]
                        if self.include_clusters:
                            row.append(cluster_feat)
                        if self.include_graph:
                            row.extend(graph_feat)
                        if self.extra_features and j > 0:
                            row.extend([
                                float(cpu[j]) - float(cpu[j - 1]),
                                float(mem[j]) - float(mem[j - 1]),
                            ])
                        elif self.extra_features:
                            row.extend([0.0, 0.0])
                        seq.append(row)
                    target = 1.0 if errs[i + self.seq_len] > 0 else 0.0
                    data.append((str(bot), seq, target))
        elif isinstance(df, list):
            df.sort(key=lambda r: r.get("ts", ""))
            by_bot: dict[str, list] = {}
            for row in df:
                b = str(row.get("bot"))
                by_bot.setdefault(b, []).append(row)
                for bot, rows in by_bot.items():
                    wf_val = float(len(str(bot)))
                    bot_id = float(abs(hash(str(bot))) % 1000) / 1000.0
                    cluster_feat = self._cluster_feature(bot)
                    graph_feat = self._graph_features(bot)
                    disks = [float(r.get("disk_io", 0.0)) for r in rows]
                    nets = [float(r.get("net_io", 0.0)) for r in rows]
                    for i in range(len(rows) - self.seq_len):
                        seq = []
                        for j, r in enumerate(rows[i : i + self.seq_len]):
                            row = [
                                float(r.get("errors", 0.0)),
                                float(r.get("cpu", 0.0)),
                                float(r.get("memory", 0.0)),
                                float(r.get("revenue", 0.0) - r.get("expense", 0.0)),
                                wf_val,
                                float(disks[i + j] + nets[i + j] - (disks[i + j - 1] + nets[i + j - 1])) if j > 0 else 0.0,
                                bot_id,
                            ]
                            if self.include_clusters:
                                row.append(cluster_feat)
                            if self.include_graph:
                                row.extend(graph_feat)
                            if self.extra_features and j > 0:
                                prev = rows[i + j - 1]
                                row.extend([
                                    float(r.get("cpu", 0.0)) - float(prev.get("cpu", 0.0)),
                                    float(r.get("memory", 0.0)) - float(prev.get("memory", 0.0)),
                                ])
                            elif self.extra_features:
                                row.extend([0.0, 0.0])
                            seq.append(row)
                        target = (
                            1.0 if float(rows[i + self.seq_len].get("errors", 0.0)) > 0 else 0.0
                        )
                        data.append((bot, seq, target))
        return data

    def _cluster_feature(self, bot: str) -> float:
        """Return cluster id feature for ``bot`` if available."""
        if not self.include_clusters or not self.error_db:
            return 0.0
        try:
            clusters = self.error_db.get_error_clusters()
            errs = self.error_db.get_bot_error_types(bot)
        except Exception:
            return 0.0
        ids = [clusters[e] for e in errs if e in clusters]
        return float(ids[0]) if ids else 0.0

    def _graph_features(self, bot: str) -> Tuple[float, float]:
        """Return average error node frequency and module count for ``bot``."""
        if not self.include_graph or not self.graph or not getattr(self.graph, "graph", None):
            return (0.0, 0.0)
        g = self.graph.graph
        bnode = f"bot:{bot}"
        if bnode not in g:
            return (0.0, 0.0)
        freqs: List[float] = []
        modules: set[str] = set()
        for enode, _, _ in g.in_edges(bnode, data=True):
            if not enode.startswith("error_type:"):
                continue
            freqs.append(
                float(g.nodes[enode].get("frequency", g.nodes[enode].get("weight", 0.0)))
            )
            for _, mnode, d in g.out_edges(enode, data=True):
                if mnode.startswith("module:"):
                    modules.add(mnode)
        avg_freq = float(sum(freqs) / len(freqs)) if freqs else 0.0
        return avg_freq, float(len(modules))

    def train(self) -> bool:
        data = self._dataset()
        if not data:
            return False
        self.default_rate = float(sum(t for _, _, t in data)) / len(data)
        if not self.use_torch:
            return True
        X = np.array([seq for _, seq, _ in data], dtype=np.float32)
        y = np.array([target for _, _, target in data], dtype=np.float32)
        x = torch.tensor(X, dtype=torch.float32)
        t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        for _ in range(self.epochs):
            if self.model_type == "transformer":
                out = self.transformer(x.transpose(0, 1))  # seq_len x batch x feat
                out = self.dropout(out)
                last = out[-1]
                logit = self.fc(last)
            else:
                out, _ = self.lstm(x)
                out = self.dropout(out)
                logit = self.fc(out[:, -1, :])
            loss = self.loss(logit, t)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return True

    # ------------------------------------------------------------------
    def _last_sequence(self, bot: str) -> List[List[float]]:
        df = self.metrics_db.fetch(self.seq_len)
        seq: List[List[float]] = []
        cluster_feat = self._cluster_feature(bot)
        graph_feat = self._graph_features(bot)
        if hasattr(df, "empty"):
            group = df[df["bot"] == bot].sort_values("ts")
            rows = group[["errors", "cpu", "memory", "disk_io", "net_io", "revenue", "expense"]].to_dict("records")
            wf_val = float(len(str(bot)))
            bot_id = float(abs(hash(str(bot))) % 1000) / 1000.0
            seq = []
            for idx, r in enumerate(rows[-self.seq_len :]):
                row = [
                    float(r["errors"]),
                    float(r["cpu"]),
                    float(r["memory"]),
                    float(r["revenue"] - r["expense"]),
                    wf_val,
                    float(r["disk_io"] + r["net_io"] - (rows[-self.seq_len + idx - 1]["disk_io"] + rows[-self.seq_len + idx - 1]["net_io"])) if idx > 0 else 0.0,
                    bot_id,
                ]
                if self.include_clusters:
                    row.append(cluster_feat)
                if self.include_graph:
                    row.extend(graph_feat)
                if self.extra_features and idx > 0:
                    prev = rows[-self.seq_len + idx - 1]
                    row.extend([
                        float(r["cpu"]) - float(prev["cpu"]),
                        float(r["memory"]) - float(prev["memory"]),
                    ])
                elif self.extra_features:
                    row.extend([0.0, 0.0])
                seq.append(row)
        elif isinstance(df, list):
            rows = [r for r in df if r.get("bot") == bot]
            rows.sort(key=lambda r: r.get("ts", ""))
            wf_val = float(len(str(bot)))
            bot_id = float(abs(hash(str(bot))) % 1000) / 1000.0
            disks = [float(r.get("disk_io", 0.0)) for r in rows]
            nets = [float(r.get("net_io", 0.0)) for r in rows]
            seq = []
            for idx, r in enumerate(rows[-self.seq_len :]):
                row = [
                    float(r.get("errors", 0.0)),
                    float(r.get("cpu", 0.0)),
                    float(r.get("memory", 0.0)),
                    float(r.get("revenue", 0.0) - r.get("expense", 0.0)),
                    wf_val,
                    float(disks[-self.seq_len + idx] + nets[-self.seq_len + idx] - (disks[-self.seq_len + idx - 1] + nets[-self.seq_len + idx - 1])) if idx > 0 else 0.0,
                    bot_id,
                ]
                if self.include_clusters:
                    row.append(cluster_feat)
                if self.include_graph:
                    row.extend(graph_feat)
                if self.extra_features and idx > 0:
                    prev = rows[-self.seq_len + idx - 1]
                    row.extend([
                        float(r.get("cpu", 0.0)) - float(prev.get("cpu", 0.0)),
                        float(r.get("memory", 0.0)) - float(prev.get("memory", 0.0)),
                    ])
                elif self.extra_features:
                    row.extend([0.0, 0.0])
                seq.append(row)
        if len(seq) < self.seq_len:
            feat_len = (
                7
                + (1 if self.include_clusters else 0)
                + (2 if self.extra_features else 0)
                + (2 if self.include_graph else 0)
            )
            pad = [[0.0] * feat_len] * (self.seq_len - len(seq))
            seq = pad + seq
        return seq

    def predict_error_prob(self, bot: str, steps: int = 1) -> List[float]:
        """Return error probabilities for ``bot`` for the next ``steps`` time steps."""
        if steps <= 0:
            return []
        seq = self._last_sequence(bot)
        wf_val = float(len(str(bot)))
        bot_id = float(abs(hash(str(bot))) % 1000) / 1000.0
        cluster_feat = self._cluster_feature(bot)
        graph_feat = self._graph_features(bot)
        preds: List[float] = []
        for _ in range(steps):
            if self.use_torch:
                with torch.no_grad():
                    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                    if self.model_type == "transformer":
                        out = self.transformer(x.transpose(0, 1))
                        out = self.dropout(out)
                        logit = self.fc(out[-1])
                    else:
                        out, _ = self.lstm(x)
                        out = self.dropout(out)
                        logit = self.fc(out[:, -1, :])
                    prob = float(torch.sigmoid(logit)[0, 0].item())
            else:
                prob = self.default_rate
            preds.append(prob)
            pad = [prob, 0.0, 0.0, 0.0, wf_val, 0.0, bot_id]
            if self.include_clusters:
                pad.append(cluster_feat)
            if self.include_graph:
                pad.extend(graph_feat)
            if self.extra_features:
                pad.extend([0.0, 0.0])
            seq = seq[1:] + [pad]
        return preds

    def predict_failure_chain(
        self, bot: str, graph: KnowledgeGraph, steps: int = 3
    ) -> List[str]:
        """Return module nodes likely affected by ``bot`` within ``steps`` hops."""

        probs = self.predict_error_prob(bot, steps=steps)
        if not probs:
            return []
        chain = graph.bot_failure_chain(bot, top=steps)
        if chain:
            return chain
        nodes = graph.cascading_effects(f"bot:{bot}", order=steps)
        return [n for n in nodes if n.startswith("module:")]

    def suggest_patches(
        self, bot: str, graph: KnowledgeGraph, top: int = 3
    ) -> List[str]:
        """Return patch candidates for ``bot`` using cluster history."""

        probs = self.predict_error_prob(bot, steps=1)
        if not probs or probs[0] <= 0.5:
            return []
        return graph.bot_patch_candidates(bot, top=top)


__all__ = ["ErrorForecaster"]
