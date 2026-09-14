from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .database import load_database

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".hxx", ".py"}
BINARY_SUFFIXES = {".ptx", ".cubin", ".fatbin", ".so", ".dll", ".lib", ".a"}
IGNORED_DIRECTORIES = {".git", "__pycache__", "node_modules", ".venv", "build", "dist"}
CUDA_HEADERS = re.compile(r"#\s*include\s*[<\"]([^>\"]*(?:cuda|cublas|cufft|curand|cusparse|cusolver|cudnn|nccl)[^>\"]*)", re.I)
KERNEL_SYNTAX = re.compile(r"<<<.*?>>>", re.S)
INLINE_PTX = re.compile(r"\basm\s*(?:volatile\s*)?\(", re.I)
CUDA_FLAGS = re.compile(r"(?:-arch=sm_|--generate-code|-gencode|--cuda-gpu-arch)", re.I)


def _source_files(root: Path) -> list[Path]:
    return [
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SOURCE_SUFFIXES | BINARY_SUFFIXES
        and not any(part in IGNORED_DIRECTORIES for part in path.relative_to(root).parts)
    ]


def analyze(root: Path, database: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"Project directory does not exist: {root}")
    entries = database or load_database()
    patterns = [(entry, re.compile(r"\b" + re.escape(entry["symbol"]) + r"\b")) for entry in entries]
    findings: list[dict[str, Any]] = []
    headers: set[str] = set()
    special: Counter[str] = Counter()
    scanned = 0
    for path in _source_files(root):
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in BINARY_SUFFIXES:
            findings.append({"kind": "binary_dependency", "file": relative, "symbol": path.name, "status": "UNKNOWN", "count": 1, "binary_format": _binary_format(path), "assessment": "requires translation" if path.suffix.lower() == ".ptx" else "contains unsupported device binary"})
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        headers.update(CUDA_HEADERS.findall(content))
        for entry, pattern in patterns:
            count = len(pattern.findall(content))
            if count:
                findings.append({"kind": "api", "file": relative, "symbol": entry["symbol"], "id": entry["id"], "status": entry["status"], "count": count, "amd_target": entry.get("amd_target")})
        for kind, pattern in (("kernel_syntax", KERNEL_SYNTAX), ("inline_ptx", INLINE_PTX), ("nvidia_compiler_flag", CUDA_FLAGS)):
            count = len(pattern.findall(content))
            if count:
                findings.append({"kind": kind, "file": relative, "symbol": kind, "status": "PARTIAL" if kind == "kernel_syntax" else "NVIDIA_SPECIFIC", "count": count})
                special[kind] += count
    api_counts = Counter()
    total = 0
    for finding in findings:
        if finding["kind"] == "api":
            api_counts[finding["status"]] += finding["count"]
            total += finding["count"]
    return {"schema_version": "1.0", "project": str(root.resolve()), "files_scanned": scanned, "cuda_headers": sorted(headers), "summary": {"cuda_api_references": total, "by_status": dict(sorted(api_counts.items())), "special_constructs": dict(sorted(special.items()))}, "findings": sorted(findings, key=lambda item: (item["file"], item["symbol"]))}


def _binary_format(path: Path) -> str:
    try:
        signature = path.read_bytes()[:4]
    except OSError:
        return "unreadable"
    if signature.startswith(b"\x7fELF"):
        return "ELF"
    if signature[:2] == b"MZ":
        return "PE/COFF"
    if path.suffix.lower() == ".ptx":
        return "PTX text"
    if path.suffix.lower() in {".cubin", ".fatbin"}:
        return "CUDA device container (unparsed)"
    return "unknown"
