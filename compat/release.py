"""Local release-readiness checks for the implemented compatibility subset."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _version(path: Path, pattern: str) -> str | None:
    match = re.search(pattern, path.read_text(encoding="utf-8"), re.M)
    return match.group(1) if match else None


def release_report(library: Path | None = None) -> dict:
    """Return factual static/package checks without claiming runtime success."""
    source_checkout = (ROOT / "pyproject.toml").is_file()
    if not source_checkout:
        from . import __version__
        return {
            "version": __version__,
            "source_checkout": False,
            "release_gate_available": False,
            "message": "release-check validates a source checkout; this installed wheel can be imported but has no release metadata tree",
            "native_library_provided": str(library) if library else None,
            "native_library_present": library.is_file() if library else None,
            "release_scope": "experimental, explicit AMD compatibility subset; not full CUDA parity",
        }
    versions = {
        "python_package": _version(ROOT / "pyproject.toml", r'^version = "([^"]+)"'),
        "python_runtime": _version(ROOT / "compat" / "__init__.py", r'^__version__ = "([^"]+)"'),
        "cmake": _version(ROOT / "CMakeLists.txt", r'project\(compatcuda VERSION ([^ )]+)'),
    }
    required = ("README.md", "CHANGELOG.md", "LICENSE", "LEGAL.md", "SECURITY.md",
                "docs/local-developer-guide.md", "docs/cuda-source-compatibility.md",
                "docs/ptx-subset.md", "docs/framework-integration.md", "docs/cuda-wheel-execution.md",
                "docs/initialization.md", "docs/site/index.html", "docs/site/README.md",
                "assets/cudatoamd-logo.svg", ".github/workflows/pages.yml")
    missing = [name for name in required if not (ROOT / name).is_file()]
    version = versions["python_package"]
    return {
        "version": version,
        "source_checkout": True,
        "release_gate_available": True,
        "version_consistent": bool(version and all(value == version for value in versions.values())),
        "versions": versions,
        "required_files_present": not missing,
        "missing_files": missing,
        "cmake_install_rules": "install(TARGETS compatcuda" in (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"),
        "c_header_smoke_test": (ROOT / "tests/native/c_header_smoke.c").is_file(),
        "native_library_provided": str(library) if library else None,
        "native_library_present": library.is_file() if library else None,
        "release_scope": "experimental, explicit AMD compatibility subset; not full CUDA parity",
    }


def release_ready(report: dict) -> bool:
    return bool(report.get("release_gate_available") and report["version_consistent"] and report["required_files_present"] and
                report["cmake_install_rules"] and report["c_header_smoke_test"])
