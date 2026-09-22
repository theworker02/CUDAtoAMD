# Buyer Demo — CUDAtoAMD

**Date:** 2026-09-21  
**Target:** fresh machine → clone → install → analyze → (optional GPU) → tests  
**Contact:** [@theworker02](https://github.com/theworker02)

Minimal path needs **Python 3.10+** only. GPU/HIP steps are optional and
hardware-dependent.

## Exact commands (CPU / tooling)

```bash
git clone https://github.com/theworker02/CUDAtoAMD.git
cd CUDAtoAMD
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
python -m compat doctor
python -m compat analyze examples
python -m compat port-plan examples
python -m compat matrix | head
python -m unittest discover -s tests -v
```

### Expected results (tooling)

- `doctor` prints host info and hipcc found/not found without crashing
- `analyze` / `port-plan` exit 0 and show inventory or an empty-findings plan
- Unit tests for analyzer/portplan/database/etc. pass without a GPU

## Optional GPU path (when ROCm/HIP is installed)

```bash
python -m compat toolchain --format json
# configure & build with COMPATCUDA_ENABLE_HIP=ON (see docs/linux-developer-guide.md
# or docs/windows-deployment.md / docs/local-developer-guide.md)
python -m pip install -e '.[inference]'
python -m compat init --library <path-to-compatcuda> --format json
python -m compat demo --operation softmax --library <path-to-compatcuda>
```

### Expected results (GPU)

- `init` reports a real AMD device name/architecture (not an NVIDIA spoof)
- `demo` compares GPU softmax output to NumPy and succeeds only on match
- Failures must be investigated honestly — absence of a GPU is not a demo pass

## Out of scope for minimal demo

- Transparent CUDA wheel execution
- Full PyTorch CUDA training
- Claiming multi-GPU / NCCL readiness
- Production SLAs or benchmarks not checked into this repository

## See also

- [`KNOWN_LIMITATIONS.md`](./KNOWN_LIMITATIONS.md)
- [`../ACQUISITION_GUIDE.md`](../ACQUISITION_GUIDE.md)
- [`../../PORTING.md`](../../PORTING.md)
