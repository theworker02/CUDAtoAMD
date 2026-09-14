"""Read-only Python wheel preflight for explicit AMD compatibility launches."""
from __future__ import annotations

import email
import hashlib
from pathlib import Path
import zipfile

MAX_ENTRIES = 10_000
MAX_UNCOMPRESSED = 256 * 1024 * 1024
CUDA_NAMES = (b"cudart", b"nvcuda", b"cublas", b"cufft", b"cudnn", b"nccl", b"nvrtc")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wheel(path: Path) -> dict:
    """Inspect archive metadata/names/import hints without extracting or executing it."""
    path = Path(path).resolve(strict=True)
    if path.suffix.lower() not in (".whl", ".zip"):
        raise ValueError("wheel preflight requires a .whl or .zip archive")
    digest = _sha256(path)
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            total = sum(item.file_size for item in entries)
            if len(entries) > MAX_ENTRIES or total > MAX_UNCOMPRESSED:
                raise ValueError("wheel archive exceeds inspection safety limits")
            names = [item.filename for item in entries]
            lower_names = [name.lower() for name in names]
            metadata = next((item for item in entries if item.filename.endswith(".dist-info/METADATA")), None)
            requirements = []
            if metadata:
                message = email.message_from_bytes(archive.read(metadata))
                requirements = message.get_all("Requires-Dist", [])
                distribution_name = message.get("Name")
                distribution_version = message.get("Version")
            else:
                distribution_name = distribution_version = None
            native = [name for name in names if Path(name).suffix.lower() in (".pyd", ".dll", ".so", ".dylib")]
            device = [name for name in names if Path(name).suffix.lower() in (".cubin", ".fatbin", ".ptx")]
            cuda_imports = []
            for item in entries:
                if Path(item.filename).suffix.lower() not in (".pyd", ".dll", ".so", ".dylib"):
                    continue
                payload = archive.read(item, min(item.file_size, 1024 * 1024)).lower()
                if any(marker in payload for marker in CUDA_NAMES):
                    cuda_imports.append(item.filename)
            nvidia_dependencies = [item for item in requirements if "nvidia-" in item.lower()]
    except zipfile.BadZipFile as error:
        raise ValueError("wheel archive is malformed") from error

    blockers = []
    if cuda_imports:
        blockers.append("native extension imports NVIDIA CUDA libraries")
    if any(Path(name).suffix.lower() in (".cubin", ".fatbin") for name in device):
        blockers.append("contains CUDA device binaries not accepted by this runtime")
    if nvidia_dependencies:
        blockers.append("declares NVIDIA runtime package dependencies")
    if blockers:
        verdict = "blocked"
        next_step = "rebuild/port the extension for the documented AMD runtime; do not spoof CUDA DLLs"
    elif native:
        verdict = "unverified_native_extension"
        next_step = "test the extension explicitly; native code is not transparently redirected"
    elif any(name.endswith(".ptx") for name in lower_names):
        verdict = "ptx_review_required"
        next_step = "extract and compile only the documented PTX subset to AMD code objects"
    else:
        verdict = "python_only_or_no_cuda_binary_detected"
        next_step = "use compat python for an explicit ROCm-enabled launch; this does not add torch.cuda support"
    return {
        "archive": str(path), "sha256": digest, "distribution_name": distribution_name,
        "distribution_version": distribution_version, "entries": len(names), "uncompressed_bytes": total,
        "native_extensions": native, "device_artifacts": device, "cuda_importing_extensions": cuda_imports,
        "nvidia_dependencies": nvidia_dependencies, "verdict": verdict, "blockers": blockers,
        "next_step": next_step,
    }


def wheel_launchable(report: dict) -> bool:
    return report["verdict"] not in ("blocked", "unverified_native_extension", "ptx_review_required")
