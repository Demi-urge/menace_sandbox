#!/usr/bin/env python3
"""Helpers for inspecting ``workflow_graph`` data.

Examples:
    python -m tools.workflow_graph_cli chain 42
    python -m tools.workflow_graph_cli diff
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dynamic_path_router import resolve_path


DEFAULT_PATH = resolve_path("sandbox_data/workflow_graph.json")


def _load_graph(path: str | Path) -> Dict[str, List[str]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}

    if isinstance(data.get("nodes"), list):
        graph: Dict[str, List[str]] = {str(n["id"]): [] for n in data.get("nodes", [])}
        for edge in data.get("edges", []):
            src = str(edge.get("source"))
            dst = str(edge.get("target"))
            graph.setdefault(src, []).append(dst)
            graph.setdefault(dst, graph.get(dst, []))
        return graph

    nodes = data.get("nodes", {})
    edges = data.get("edges", {})
    graph = {str(n): [str(d) for d in edges.get(n, {})] for n in nodes}
    for n in nodes:
        graph.setdefault(str(n), graph.get(str(n), []))
    return graph


def _load_sets(path: str | Path) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return set(), set()

    if isinstance(data.get("nodes"), list):
        nodes = {str(n["id"]) for n in data.get("nodes", [])}
        edges = {(str(e["source"]), str(e["target"])) for e in data.get("edges", [])}
    else:
        nodes = {str(n) for n in data.get("nodes", {})}
        edges = {
            (str(src), str(dst))
            for src, dsts in data.get("edges", {}).items()
            for dst in dsts
        }
    return nodes, edges


def _dependency_chains(graph: Dict[str, List[str]], start: str) -> List[List[str]]:
    chains: List[List[str]] = []

    def dfs(path: List[str]) -> None:
        last = path[-1]
        children = graph.get(last, [])
        if not children:
            chains.append(path)
            return
        for nxt in children:
            if nxt in path:
                continue
            dfs(path + [nxt])

    if start not in graph:
        return []
    dfs([start])
    return chains


def _latest_snapshots(base_path: str | Path) -> List[str]:
    base, _ = os.path.splitext(str(base_path))
    pattern = f"{base}.*.json"
    return sorted(glob.glob(pattern))


def cmd_chain(args: argparse.Namespace) -> None:
    path = resolve_path(args.path)
    graph = _load_graph(path)
    chains = _dependency_chains(graph, str(args.workflow_id))
    print(json.dumps(chains))


def cmd_diff(args: argparse.Namespace) -> None:
    path = resolve_path(args.path)
    snaps = _latest_snapshots(path)
    if len(snaps) < 2:
        print(json.dumps({"error": "not enough snapshots"}))
        return
    old_path, new_path = snaps[-2], snaps[-1]
    old_nodes, old_edges = _load_sets(old_path)
    new_nodes, new_edges = _load_sets(new_path)
    diff = {
        "from": old_path,
        "to": new_path,
        "added_nodes": sorted(new_nodes - old_nodes),
        "removed_nodes": sorted(old_nodes - new_nodes),
        "added_edges": sorted([list(e) for e in new_edges - old_edges]),
        "removed_edges": sorted([list(e) for e in old_edges - new_edges]),
    }
    print(json.dumps(diff))


def main() -> None:
    parser = argparse.ArgumentParser(description="Workflow graph utilities")
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help="Path to workflow_graph.json",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    chain_p = sub.add_parser("chain", help="print dependency chains for a workflow")
    chain_p.add_argument("workflow_id")

    sub.add_parser(
        "diff", help="show recent changes compared to previous saved versions"
    )

    args = parser.parse_args()
    if args.cmd == "chain":
        cmd_chain(args)
    elif args.cmd == "diff":
        cmd_diff(args)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

