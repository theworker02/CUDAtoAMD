"""Build a prioritized, human-readable porting plan from an analyze() report.

This module does **not** rewrite source or invoke HIPIFY. It ranks findings so
developers can decide what to address first. Effort hints are heuristics based
on compatibility-database status, not measured engineering estimates.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# Highest-priority statuses first: blockers before easy mappings.
STATUS_PRIORITY: tuple[str, ...] = (
    "UNSUPPORTED",
    "NVIDIA_SPECIFIC",
    "UNKNOWN",
    "PARTIAL",
    "ADAPTED",
    "DIRECT",
)

_EFFORT_BY_STATUS: dict[str, str] = {
    "UNSUPPORTED": "high",
    "NVIDIA_SPECIFIC": "high",
    "UNKNOWN": "high",
    "PARTIAL": "medium",
    "ADAPTED": "medium",
    "DIRECT": "low",
}

_ACTION_BY_STATUS: dict[str, str] = {
    "UNSUPPORTED": (
        "No documented AMD mapping in the compatibility inventory. Redesign, "
        "remove, or isolate this usage behind a platform abstraction before porting."
    ),
    "NVIDIA_SPECIFIC": (
        "NVIDIA-specific construct or symbol. Replace with a portable or HIP-native "
        "equivalent; do not expect HIPIFY to produce a correct result automatically."
    ),
    "UNKNOWN": (
        "Not classified or not inventory-backed (e.g. device binaries). Inspect "
        "manually; PTX may use `compat ptx-object` only for the documented subset."
    ),
    "PARTIAL": (
        "Partial mapping exists. Validate semantics and capability gates on target "
        "hardware before treating as equivalent."
    ),
    "ADAPTED": (
        "Documented adapted mapping (argument/error/library differences). Port via "
        "HIPIFY or manual rewrite, then conformance-test on AMD."
    ),
    "DIRECT": (
        "Documented direct HIP analogue. Candidate for HIPIFY or manual rename; "
        "still verify with functional tests—DIRECT is not an automatic correctness guarantee."
    ),
}

_CAVEATS: list[str] = [
    "This plan is advisory inventory prioritization, not automatic migration.",
    "DIRECT and ADAPTED statuses describe database mappings, not proven ABI drop-in success.",
    "HIPIFY should be run in a separate working tree; preserve original CUDA sources.",
    "Precompiled CUDA device binaries (.cubin, .fatbin) are not translated by this project.",
    "PTX support is limited to the documented validated subset (`compat ptx-object`).",
    "Transparent CUDA-wheel / torch.cuda execution is not implemented.",
    "Effort hints (low/medium/high) are status-based heuristics, not measured person-days.",
]


def _status_rank(status: str) -> int:
    try:
        return STATUS_PRIORITY.index(status)
    except ValueError:
        return len(STATUS_PRIORITY)


def _effort_for_finding(finding: dict[str, Any]) -> str:
    status = finding.get("status", "UNKNOWN")
    kind = finding.get("kind", "")
    if kind == "binary_dependency":
        return "high"
    if kind in {"inline_ptx", "nvidia_compiler_flag"}:
        return "high"
    if kind == "kernel_syntax":
        return "medium"
    return _EFFORT_BY_STATUS.get(status, "high")


def _action_for_finding(finding: dict[str, Any]) -> str:
    kind = finding.get("kind", "api")
    status = finding.get("status", "UNKNOWN")
    if kind == "binary_dependency":
        fmt = finding.get("binary_format", "unknown")
        if finding.get("symbol", "").endswith(".ptx") or fmt == "PTX text":
            return (
                "Review PTX against the documented subset; compile only supported "
                "constructs with `compat ptx-object`, otherwise rewrite as HIP/source kernels."
            )
        return (
            f"CUDA/device binary ({fmt}) cannot be executed or translated as-is. "
            "Rebuild from source for AMD (HIP/hsaco) or exclude from the AMD build."
        )
    if kind == "kernel_syntax":
        return (
            "CUDA launch syntax (`<<<>>>`) requires hipcc/Clang CUDA-language lowering "
            "or conversion to hipLaunchKernelGGL / explicit AMD module launch."
        )
    if kind == "inline_ptx":
        return (
            "Inline PTX/asm is NVIDIA-specific. Rewrite as HIP device code or the "
            "documented PTX subset path; do not assume silent translation."
        )
    if kind == "nvidia_compiler_flag":
        return (
            "NVIDIA compiler flags (-arch=sm_*, -gencode, etc.) must be replaced with "
            "HIP/AMD architecture flags (e.g. --offload-arch=gfxXXXX)."
        )
    return _ACTION_BY_STATUS.get(status, _ACTION_BY_STATUS["UNKNOWN"])


def _worst_effort(efforts: set[str]) -> str:
    order = {"high": 0, "medium": 1, "low": 2}
    return min(efforts, key=lambda e: order.get(e, 0))


def build_port_plan(report: dict[str, Any]) -> dict[str, Any]:
    """Produce a prioritized porting plan dict from an ``analyze()`` report."""
    findings = list(report.get("findings") or [])
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_file: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for finding in findings:
        status = finding.get("status", "UNKNOWN")
        by_status[status].append(finding)
        per_file[finding.get("file", "<unknown>")].append(finding)

    prioritized_groups: list[dict[str, Any]] = []
    for status in STATUS_PRIORITY:
        group_findings = by_status.get(status)
        if not group_findings:
            continue
        prioritized_groups.append(
            {
                "status": status,
                "count": sum(int(f.get("count", 1)) for f in group_findings),
                "finding_count": len(group_findings),
                "effort_hint": _EFFORT_BY_STATUS.get(status, "high"),
                "recommended_action": _ACTION_BY_STATUS.get(status, _ACTION_BY_STATUS["UNKNOWN"]),
                "findings": sorted(
                    group_findings,
                    key=lambda f: (f.get("file", ""), f.get("symbol", "")),
                ),
            }
        )

    # Include any unexpected statuses after the known order.
    for status in sorted(by_status.keys()):
        if status in STATUS_PRIORITY:
            continue
        group_findings = by_status[status]
        prioritized_groups.append(
            {
                "status": status,
                "count": sum(int(f.get("count", 1)) for f in group_findings),
                "finding_count": len(group_findings),
                "effort_hint": "high",
                "recommended_action": _ACTION_BY_STATUS["UNKNOWN"],
                "findings": sorted(
                    group_findings,
                    key=lambda f: (f.get("file", ""), f.get("symbol", "")),
                ),
            }
        )

    file_actions: list[dict[str, Any]] = []
    for path in sorted(per_file.keys()):
        file_findings = per_file[path]
        statuses = sorted({f.get("status", "UNKNOWN") for f in file_findings}, key=_status_rank)
        efforts = {_effort_for_finding(f) for f in file_findings}
        # Deduplicate recommended actions while preserving severity order.
        actions: list[str] = []
        seen: set[str] = set()
        for finding in sorted(file_findings, key=lambda f: (_status_rank(f.get("status", "UNKNOWN")), f.get("symbol", ""))):
            action = _action_for_finding(finding)
            if action not in seen:
                seen.add(action)
                actions.append(action)
        symbols = sorted({f.get("symbol", "") for f in file_findings if f.get("symbol")})
        file_actions.append(
            {
                "file": path,
                "statuses": statuses,
                "effort_hint": _worst_effort(efforts),
                "symbol_count": len(symbols),
                "symbols": symbols,
                "recommended_actions": actions,
                "finding_count": len(file_findings),
            }
        )

    # Files with UNSUPPORTED/NVIDIA_SPECIFIC first, then UNKNOWN, etc.
    file_actions.sort(
        key=lambda item: (
            min(_status_rank(s) for s in item["statuses"]),
            0 if item["effort_hint"] == "high" else 1 if item["effort_hint"] == "medium" else 2,
            item["file"],
        )
    )

    summary = report.get("summary") or {}
    return {
        "schema_version": "1.0",
        "kind": "port_plan",
        "project": report.get("project"),
        "files_scanned": report.get("files_scanned", 0),
        "cuda_headers": list(report.get("cuda_headers") or []),
        "summary": {
            "cuda_api_references": summary.get("cuda_api_references", 0),
            "by_status": dict(summary.get("by_status") or {}),
            "special_constructs": dict(summary.get("special_constructs") or {}),
            "files_with_findings": len(file_actions),
            "total_findings": len(findings),
        },
        "priority_order": list(STATUS_PRIORITY),
        "groups": prioritized_groups,
        "per_file": file_actions,
        "caveats": list(_CAVEATS),
        "next_steps": [
            "Address UNSUPPORTED and NVIDIA_SPECIFIC findings (or isolate them) before HIPIFY.",
            "Clarify UNKNOWN items (binaries, untracked symbols) with manual review.",
            "For PARTIAL/ADAPTED/DIRECT, plan HIPIFY or manual rewrite in a separate tree.",
            "Build with hipcc, run functional tests, and record device/ROCm/compiler/OS versions.",
            "Re-run `compat analyze` / `compat port-plan` after each major change.",
        ],
    }


def render_port_plan(plan: dict[str, Any]) -> str:
    """Render a port plan as plain text for CLI output."""
    lines: list[str] = [
        "CUDA-to-AMD prioritized porting plan",
        "",
        f"Project: {plan.get('project')}",
        f"Files scanned: {plan.get('files_scanned', 0)}",
        f"Files with findings: {plan['summary'].get('files_with_findings', 0)}",
        f"Total findings: {plan['summary'].get('total_findings', 0)}",
        f"CUDA API references: {plan['summary'].get('cuda_api_references', 0)}",
    ]

    by_status = plan["summary"].get("by_status") or {}
    if by_status:
        lines.append("")
        lines.append("API references by status:")
        for status, count in sorted(by_status.items(), key=lambda kv: _status_rank(kv[0])):
            lines.append(f"  {status}: {count}")

    special = plan["summary"].get("special_constructs") or {}
    if special:
        lines.append("")
        lines.append("Special constructs:")
        for kind, count in sorted(special.items()):
            lines.append(f"  {kind}: {count}")

    headers = plan.get("cuda_headers") or []
    if headers:
        lines.append("")
        lines.append("CUDA headers:")
        for header in headers:
            lines.append(f"  {header}")

    lines.extend(["", "Prioritized groups (blockers first):"])
    groups = plan.get("groups") or []
    if not groups:
        lines.append("  (no findings — nothing to prioritize)")
    for group in groups:
        lines.append("")
        lines.append(
            f"  [{group['status']}] findings={group['finding_count']} "
            f"refs={group['count']} effort={group['effort_hint']}"
        )
        lines.append(f"    Action: {group['recommended_action']}")
        # Show a compact sample of symbols (cap for readability).
        sample = group.get("findings") or []
        for finding in sample[:12]:
            target = finding.get("amd_target")
            target_note = f" -> {target}" if target else ""
            lines.append(
                f"    - {finding.get('file')}: {finding.get('symbol')} "
                f"(×{finding.get('count', 1)}, {finding.get('kind', 'api')}){target_note}"
            )
        if len(sample) > 12:
            lines.append(f"    ... and {len(sample) - 12} more in this group")

    lines.extend(["", "Per-file recommended actions:"])
    per_file = plan.get("per_file") or []
    if not per_file:
        lines.append("  (no files with findings)")
    for item in per_file:
        lines.append("")
        lines.append(
            f"  {item['file']}  [{', '.join(item['statuses'])}]  effort={item['effort_hint']}"
        )
        lines.append(f"    Symbols ({item['symbol_count']}): {', '.join(item['symbols'][:20])}")
        if item["symbol_count"] > 20:
            lines.append(f"    ... and {item['symbol_count'] - 20} more symbols")
        for action in item["recommended_actions"]:
            lines.append(f"    - {action}")

    lines.extend(["", "Caveats:"])
    for caveat in plan.get("caveats") or []:
        lines.append(f"  - {caveat}")

    lines.extend(["", "Suggested next steps:"])
    for step in plan.get("next_steps") or []:
        lines.append(f"  1. {step}" if False else f"  - {step}")

    return "\n".join(lines)
