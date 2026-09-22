from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .analyzer import analyze
from .database import default_database_path, load_database
from .matrix import render_matrix, write_matrix
from .portplan import build_port_plan, render_port_plan


def _text_report(report: dict) -> str:
    summary = report["summary"]
    lines = ["CUDA-to-AMD compatibility analysis", "", f"Project: {report['project']}", f"Files scanned: {report['files_scanned']}", f"CUDA API references: {summary['cuda_api_references']}"]
    for status, count in summary["by_status"].items():
        lines.append(f"  {status}: {count}")
    if report["cuda_headers"]:
        lines.extend(["", "CUDA headers:"] + [f"  {header}" for header in report["cuda_headers"]])
    if summary["special_constructs"]:
        lines.extend(["", "Special constructs:"] + [f"  {kind}: {count}" for kind, count in summary["special_constructs"].items()])
    return "\n".join(lines)


def _doctor(include_cuda_api: bool = False) -> str:
    entries = load_database()
    hipcc = _find_hipcc()
    lines = ["CUDA-to-AMD Compatibility Runtime", "", "Host", f"  OS: {platform.system()} {platform.release()}", f"  Architecture: {platform.machine()}", "", "Toolchain", f"  hipcc: {hipcc or 'not found'}", "  GPU execution: not probed by doctor; run native CTest for validation", "", "Compatibility database", f"  path: {default_database_path()}", f"  tracked interfaces: {len(entries)}"]
    if include_cuda_api:
        from collections import Counter
        counts = Counter(entry["status"] for entry in entries)
        lines.extend(["", "CUDA API inventory (tracked interfaces, not a coverage percentage)"] + [f"  {status}: {count}" for status, count in sorted(counts.items())])
    lines.extend(["", "Status: partial Runtime/Driver core, native modules, mixed-precision GEMM and optional complex 1D FFT implemented. Native graph capture/replay and explicit CPU-staged inference are available; full CUDA parity is not."])
    return "\n".join(lines)


def _find_hipcc() -> str | None:
    resolved = shutil.which("hipcc")
    if resolved:
        return resolved
    system = platform.system()
    if system == "Windows":
        root = Path(os.environ.get("ProgramFiles", r"C:\\Program Files")) / "AMD" / "ROCm"
        candidates = sorted(root.glob("*/bin/hipcc.exe"), reverse=True)
        return str(candidates[0]) if candidates else None
    # Linux (and other Unix): common ROCm install layouts
    candidates: list[Path] = [
        Path("/opt/rocm/bin/hipcc"),
    ]
    rocm_path = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
    if rocm_path:
        candidates.insert(0, Path(rocm_path) / "bin" / "hipcc")
    # Versioned installs under /opt/rocm-*
    versioned = sorted(Path("/opt").glob("rocm*/bin/hipcc"), reverse=True)
    candidates.extend(versioned)
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    return None


def _toolchain_report() -> dict:
    hipcc = _find_hipcc()
    root = Path(hipcc).parents[1] if hipcc else None
    hipify = root / "bin" / ("hipify-clang.exe" if platform.system() == "Windows" else "hipify-clang") if root else None
    return {
        "hipcc": hipcc,
        "hipify_clang": str(hipify) if hipify and hipify.is_file() else None,
        "cuda_source_compiler": "compat code-object (HIP-Clang AMD code-object output)",
        "ptx_compiler": "compat ptx-object (documented validated subset)",
        "nvcc_detected": shutil.which("nvcc"),
        "nvcc_usage": "not used to target AMD hardware",
        "ready": bool(hipcc),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compat", description="CUDA-to-AMD compatibility tools")
    commands = parser.add_subparsers(dest="command", required=True)
    ptx_parser = commands.add_parser("ptx-object", help="Experimental validated PTX subset to native AMD ELF")
    ptx_parser.add_argument("source", type=Path)
    ptx_parser.add_argument("-o", "--output", type=Path, required=True)
    ptx_parser.add_argument("--arch", required=True)
    object_parser=commands.add_parser("code-object",help="Compile CUDA-syntax kernel sources to explicit AMD ELF (not PTX translation)")
    object_parser.add_argument("source",type=Path)
    object_parser.add_argument("-o","--output",type=Path,required=True)
    object_parser.add_argument("--arch",required=True)
    analyze_parser = commands.add_parser("analyze", help="Read-only CUDA-oriented project analysis")
    analyze_parser.add_argument("project", type=Path)
    analyze_parser.add_argument("--format", choices=("text", "json"), default="text")
    port_plan_parser = commands.add_parser(
        "port-plan",
        help="Prioritized porting plan from analyze() findings (advisory; not migration)",
    )
    port_plan_parser.add_argument("project", type=Path)
    port_plan_parser.add_argument("--format", choices=("text", "json"), default="text")
    port_plan_parser.add_argument("--output", type=Path, help="Write plan to this file instead of stdout")
    matrix_parser = commands.add_parser("matrix", help="Generate or print COMPATIBILITY.md")
    matrix_parser.add_argument("--output", type=Path)
    doctor_parser = commands.add_parser("doctor", help="Report local analyzer/toolchain availability")
    doctor_parser.add_argument("--cuda-api", action="store_true", help="Include compatibility inventory counts")
    doctor_parser.add_argument("--native", action="store_true", help="Load the native library and query actual devices")
    doctor_parser.add_argument("--library", type=Path, help="Explicit compatcuda library path")
    init_parser = commands.add_parser("init", help="Initialize the real HIP/ROCm AMD backend")
    init_parser.add_argument("--library", type=Path, help="Explicit compatcuda native library")
    init_parser.add_argument("--device", type=int, default=0)
    init_parser.add_argument("--self-test", action="store_true", help="Run the native softmax numerical self-test after initialization")
    init_parser.add_argument("--module", type=Path, help="Trusted neural.hsaco for --self-test")
    init_parser.add_argument("--format", choices=("text", "json"), default="text")
    toolchain_parser = commands.add_parser("toolchain", help="Report CUDA-facing and HIP compiler tooling")
    toolchain_parser.add_argument("--format", choices=("text", "json"), default="text")
    demo_parser = commands.add_parser("demo", help="Run a neural kernel on AMD and compare every result with NumPy")
    demo_parser.add_argument("--operation", choices=("rmsnorm","softmax","swiglu","rope"), default="softmax")
    demo_parser.add_argument("--library", type=Path)
    demo_parser.add_argument("--module", type=Path, help="Trusted neural.hsaco built for this GPU")
    demo_parser.add_argument("--device", type=int, default=0)
    profile_parser = commands.add_parser("profile", help="Report a named compatibility profile")
    profile_parser.add_argument("name", choices=("ai",))
    framework_parser = commands.add_parser("framework-doctor", help="Report the explicit PyTorch AMD inference integration")
    framework_parser.add_argument("--format", choices=("text", "json"), default="text")
    release_parser = commands.add_parser("release-check", help="Check local release metadata and package readiness")
    release_parser.add_argument("--library", type=Path, help="Optionally require this built native library")
    release_parser.add_argument("--format", choices=("text", "json"), default="text")
    wheel_parser = commands.add_parser("wheel-doctor", help="Read-only CUDA-wheel/native-extension preflight")
    wheel_parser.add_argument("wheel", type=Path)
    wheel_parser.add_argument("--format", choices=("text", "json"), default="text")
    python_parser = commands.add_parser("python", help="Run a Python script with an explicit ROCm environment")
    python_parser.add_argument("script", type=Path)
    python_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    python_parser.add_argument("--wheel", type=Path, help="Require a non-blocked preflight for this wheel")
    python_parser.add_argument("--wheel-sha256", help="Require this lowercase SHA-256 for --wheel")
    python_parser.add_argument("--library", type=Path, help="Set AMD_RUNTIME_LIB_PATH for explicit native bindings")
    python_parser.add_argument("--dry-run", action="store_true")
    run_parser = commands.add_parser("run", help="Launch explicitly against the compatibility environment")
    run_parser.add_argument("program", type=Path)
    run_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    run_parser.add_argument("--dry-run", action="store_true")
    cc_parser = commands.add_parser("cc", help="Run the explicit HIPIFY → hipcc source pipeline")
    cc_parser.add_argument("source", type=Path)
    cc_parser.add_argument("-o", "--output", required=True, type=Path)
    cc_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command in ("code-object", "ptx-object"):
        try:
            from .compiler import compile_code_object
            from .ptx import compile_ptx
            compiler=_find_hipcc()
            if not compiler:raise RuntimeError("HIP compiler not found; install/configure the HIP SDK")
            compile_fn = compile_ptx if args.command == "ptx-object" else compile_code_object
            return compile_fn(args.source,args.output,args.arch,compiler=compiler)
        except (OSError,ValueError,RuntimeError) as error:
            print(f"Code-object compilation failed: {error}")
            return 1
    if args.command == "demo" or args.command == "init" or (args.command == "doctor" and args.native):
        try:
            from .local import demo, initialize, inspect_runtime
            if args.command == "demo": result = demo(args.operation,args.library,args.device,args.module)
            elif args.command == "init": result = initialize(args.library,args.device,args.self_test,args.module)
            else: result = inspect_runtime(args.library)
            print(json.dumps(result,indent=2) if getattr(args, "format", "json") == "json" else _mapping_text(result))
            return 0
        except ImportError as error:
            print(f"Optional dependency missing: {error}. Install with: python -m pip install '.[inference]'")
            return 1
        except Exception as error:
            print(f"Local runtime check failed: {error}")
            return 1
    if args.command == "analyze":
        report = analyze(args.project)
        print(json.dumps(report, indent=2) if args.format == "json" else _text_report(report))
    elif args.command == "port-plan":
        try:
            report = analyze(args.project)
            plan = build_port_plan(report)
            payload = json.dumps(plan, indent=2) if args.format == "json" else render_port_plan(plan)
            if args.output:
                args.output.write_text(payload + ("\n" if not payload.endswith("\n") else ""), encoding="utf-8")
                print(f"Wrote {args.output}")
            else:
                print(payload)
        except (OSError, ValueError) as error:
            print(f"Port plan failed: {error}")
            return 1
    elif args.command == "matrix":
        entries = load_database()
        if args.output:
            write_matrix(entries, args.output)
            print(f"Wrote {args.output}")
        else:
            print(render_matrix(entries))
    elif args.command == "profile":
        print(_ai_profile())
    elif args.command == "toolchain":
        report = _toolchain_report()
        print(json.dumps(report, indent=2) if args.format == "json" else _mapping_text(report))
    elif args.command == "framework-doctor":
        from .framework import framework_report
        report = framework_report()
        if args.format == "json":
            print(json.dumps(report, indent=2))
        else:
            print("PyTorch AMD inference integration")
            for key, value in report.items(): print(f"  {key}: {value}")
    elif args.command == "release-check":
        from .release import release_ready, release_report
        report = release_report(args.library)
        if args.format == "json":
            print(json.dumps(report, indent=2))
        else:
            print("Release readiness")
            for key, value in report.items(): print(f"  {key}: {value}")
        return 0 if release_ready(report) and (args.library is None or report["native_library_present"]) else 1
    elif args.command == "wheel-doctor":
        try:
            from .wheel import inspect_wheel
            report = inspect_wheel(args.wheel)
            if args.format == "json": print(json.dumps(report, indent=2))
            else:
                print("CUDA wheel preflight")
                for key, value in report.items(): print(f"  {key}: {value}")
            return 0 if not report["blockers"] else 2
        except (OSError, ValueError) as error:
            print(f"Wheel preflight failed: {error}")
            return 1
    elif args.command == "python":
        try:
            return _python(args.script, args.arguments, args.wheel, args.wheel_sha256, args.library, args.dry_run)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Explicit Python launch failed: {error}")
            return 1
    elif args.command == "run":
        return _run(args.program, args.arguments, args.dry_run)
    elif args.command == "cc":
        return _compile(args.source, args.output, args.arguments)
    else:
        print(_doctor(include_cuda_api=args.cuda_api))
    return 0


def _mapping_text(value: dict) -> str:
    return "\n".join(f"{key}: {item}" for key, item in value.items())


def _rocm_root() -> Path | None:
    hipcc = _find_hipcc()
    return Path(hipcc).parents[1] if hipcc else None


def _run(program: Path, arguments: list[str], dry_run: bool) -> int:
    if not program.is_file():
        raise ValueError(f"Program does not exist: {program}")
    root = _rocm_root()
    if not root:
        raise RuntimeError("ROCm HIP SDK was not found")
    environment = os.environ.copy()
    environment["PATH"] = str(root / "bin") + os.pathsep + environment.get("PATH", "")
    command = [str(program.resolve()), *arguments]
    print("Compatibility launch (explicit; no DLL injection)")
    print("Command: " + " ".join(command))
    if dry_run:
        return 0
    return subprocess.run(command, env=environment, check=False).returncode


def _python(script: Path, arguments: list[str], wheel: Path | None, wheel_sha256: str | None, library: Path | None, dry_run: bool) -> int:
    if not script.is_file(): raise ValueError(f"Python script does not exist: {script}")
    if wheel_sha256 and not wheel: raise ValueError("--wheel-sha256 requires --wheel")
    if wheel:
        from .wheel import inspect_wheel, wheel_launchable
        report = inspect_wheel(wheel)
        if wheel_sha256 and (not re.fullmatch(r"[0-9a-f]{64}", wheel_sha256) or report["sha256"] != wheel_sha256):
            print("CUDA wheel preflight blocked launch: wheel SHA-256 does not match the approved artifact")
            return 2
        if not wheel_launchable(report):
            print("CUDA wheel preflight blocked launch: " + "; ".join(report["blockers"] or [report["verdict"]]))
            print("Next step: " + report["next_step"])
            return 2
    root = _rocm_root()
    if not root: raise RuntimeError("ROCm HIP SDK was not found")
    environment = os.environ.copy()
    environment["PATH"] = str(root / "bin") + os.pathsep + environment.get("PATH", "")
    if library:
        if not library.is_file(): raise ValueError(f"Native library does not exist: {library}")
        environment["AMD_RUNTIME_LIB_PATH"] = str(library.resolve())
    command = [sys.executable, str(script.resolve()), *arguments]
    print("Explicit AMD Python launch (no CUDA DLL spoofing or extension interception)")
    print("Command: " + " ".join(command))
    if dry_run: return 0
    return subprocess.run(command, env=environment, check=False).returncode


def _compile(source: Path, output: Path, arguments: list[str]) -> int:
    if not source.is_file():
        raise ValueError(f"CUDA source does not exist: {source}")
    root = _rocm_root()
    if not root:
        raise RuntimeError("ROCm HIP SDK was not found")
    suffix = ".exe" if platform.system() == "Windows" else ""
    hipify = root / "bin" / f"hipify-clang{suffix}"
    hipcc = root / "bin" / f"hipcc{suffix}"
    if not hipify.is_file():
        raise RuntimeError(f"hipify-clang not found at {hipify}")
    if not hipcc.is_file():
        raise RuntimeError(f"hipcc not found at {hipcc}")
    translated = output.with_suffix(".hip.cpp")
    translated.parent.mkdir(parents=True, exist_ok=True)
    # HIPIFY owns AST-aware CUDA→HIP translation; retain original source and emit a separate artifact.
    first = subprocess.run([str(hipify), str(source), "-o", str(translated)], check=False)
    if first.returncode:
        return first.returncode
    return subprocess.run([str(hipcc), str(translated), "-o", str(output), *arguments], check=False).returncode


def _ai_profile() -> str:
    hipcc = _find_hipcc()
    status = "DETECTED" if hipcc else "UNAVAILABLE"
    return "\n".join([
        "AI Compatibility Environment", "", "Runtime",
        f"  HIP compiler: {status}",
        "  CUDA-facing core dispatch: IMPLEMENTED (requires HIP-enabled build and runtime validation)",
        "", "Math", "  FP32/FP16/BF16 input GEMM, FP32 output: IMPLEMENTED (HIP build)", "",
        "  Complex FP32 1D FFT: IMPLEMENTED (optional hipFFT build)", "  BLASLt / RAND / SPARSE / SOLVER: NOT IMPLEMENTED", "",
        "Neural network", "  FP32 RMSNorm / softmax / SwiGLU / RoPE: IMPLEMENTED (native code object required)", "",
        "Distributed", "  Collectives: WINDOWS_BLOCKED", "",
        "Framework", "  PyTorch: explicit CPU float32 inference bridge; no autograd or CUDA-wheel interception", "",
        "Training readiness", "  SINGLE GPU: NOT READY", "  MULTI GPU: NOT READY"
    ])


if __name__ == "__main__":
    raise SystemExit(main())
