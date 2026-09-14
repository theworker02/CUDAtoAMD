"""Explicit CUDA-kernel-subset -> HIP-Clang -> native AMD code object."""
import os
from pathlib import Path
import re
import subprocess
import tempfile


def compile_code_object(source, output, arch, *, compiler):
    source=Path(source).resolve(strict=True)
    output=Path(output).resolve()
    if source==output or output.exists():
        raise ValueError("output must be a new path, different from the source")
    if source.suffix not in (".cu",".hip",".cpp"):
        raise ValueError("expected a .cu, .hip or .cpp kernel source")
    if not re.fullmatch(r"gfx[0-9a-f]+(?::(?:xnack|sramecc)[+-])*",arch):
        raise ValueError("arch must be an explicit AMD gfx target, not an NVIDIA sm target")
    if not output.parent.is_dir():
        raise ValueError("output directory must exist")
    # Compile into an isolated staging file; never clobber an existing output.
    with tempfile.TemporaryDirectory(prefix="compatcc-",dir=output.parent) as temporary:
        artifact=Path(temporary)/"kernel.hsaco"
        command=[str(compiler),"-x","hip","-include","hip/hip_runtime.h","--genco","--no-gpu-bundle-output",
                 "--offload-arch="+arch,str(source),"-o",str(artifact)]
        result=subprocess.run(command,check=False)
        if result.returncode:return result.returncode
        with artifact.open("rb") as stream:
            if stream.read(4)!=b"\x7fELF":
                raise RuntimeError("compiler did not produce an unbundled ELF code object")
        # Hard-link creation is atomic and refuses a concurrently created output.
        os.link(artifact,output)
    return 0
