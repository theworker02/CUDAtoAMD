"""Generate a checked-in ABI inventory from the project-owned API specification."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = json.loads((ROOT / "spec" / "api.json").read_text(encoding="utf-8"))
lines = ["# ABI Symbol Inventory", "", "Generated from `spec/api.json`. Symbols are declarations, not a claim of all symbols being executable.", ""]
for library in spec["libraries"]:
    lines.extend([f"## `{library['header']}`", "", "| Symbol | ABI status |", "| --- | --- |"])
    lines.extend(f"| `{symbol}` | declared; backend-dependent |" for symbol in library["symbols"])
    lines.append("")
(ROOT / "docs" / "abi-symbols.md").write_text("\n".join(lines), encoding="utf-8")
