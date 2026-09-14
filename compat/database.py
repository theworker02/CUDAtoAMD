from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VALID_STATUSES = {"DIRECT", "ADAPTED", "TRANSLATED", "EMULATED", "PARTIAL", "REMOTE_CAPABLE", "UNSUPPORTED", "NVIDIA_SPECIFIC", "UNKNOWN"}
REQUIRED_FIELDS = {"id", "symbol", "namespace", "category", "status", "semantic_compatibility", "translation_strategy"}


def default_database_path() -> Path:
    return Path(__file__).parents[1] / "compatibility" / "cuda_api.json"


def load_database(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or default_database_path()
    entries = json.loads(target.read_text(encoding="utf-8"))
    validate_database(entries)
    return entries


def validate_database(entries: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    symbols: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"Entry {index} is missing required fields: {', '.join(sorted(missing))}")
        if entry["status"] not in VALID_STATUSES:
            raise ValueError(f"{entry['id']}: invalid status {entry['status']}")
        if entry["id"] in ids or (entry["namespace"], entry["symbol"]) in symbols:
            raise ValueError(f"Duplicate compatibility entry: {entry['id']}")
        ids.add(entry["id"])
        symbols.add((entry["namespace"], entry["symbol"]))
