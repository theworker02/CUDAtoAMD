from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def render_matrix(entries: list[dict[str, Any]]) -> str:
    counts = Counter(entry["status"] for entry in entries)
    lines = ["# Compatibility Matrix", "", "Generated from `compatibility/cuda_api.json`; this is an interface inventory, not an application-success percentage.", "", "## Inventory statistics", "", "| Status | Interfaces |", "| --- | ---: |"]
    for status, count in sorted(counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend([f"| **Total tracked interfaces** | **{len(entries)}** |", "", "A `DIRECT` entry has a documented HIP/ROCm analogue; it is not an assertion that ABI-level CUDA drop-in execution already exists. `PARTIAL` and `ADAPTED` entries must be validated by conformance tests before runtime support is claimed.", ""])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[entry["category"]].append(entry)
    for category in sorted(groups):
        lines.extend([f"## {category.title()}", "", "| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |", "| --- | --- | --- | --- | --- | --- |"])
        for entry in sorted(groups[category], key=lambda x: x["symbol"]):
            lines.append("| {symbol} | {namespace} | {status} | {target} | {semantic} | {strategy} |".format(symbol=entry["symbol"], namespace=entry["namespace"], status=entry["status"], target=entry.get("amd_target", "—"), semantic=entry["semantic_compatibility"], strategy=entry["translation_strategy"]))
        lines.append("")
    return "\n".join(lines)


def write_matrix(entries: list[dict[str, Any]], output: Path) -> None:
    output.write_text(render_matrix(entries), encoding="utf-8")
