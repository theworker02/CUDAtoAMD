# Local release checklist

This project is released only for its documented experimental AMD compatibility subset. Do not describe a release as CUDA parity, a CUDA-wheel runner, or a replacement GPU driver.

From a clean source checkout with the Windows HIP SDK and Visual Studio Build Tools available:

```powershell
cmake -S . -B build-release -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCOMPATCUDA_ENABLE_HIP=ON -DCOMPATCUDA_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\7.1"
cmake --build build-release
ctest --test-dir build-release --output-on-failure
python -m unittest discover -s tests -v
python -m compat release-check --library build-release/compatcuda.dll --format json
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
cmake --install build-release --prefix staging
```

`release-check` checks repository metadata, required user documentation, CMake install rules and C-header coverage. It is deliberately a **source-checkout** gate; an installed wheel reports that the gate is unavailable rather than failing or pretending to validate packaged native GPU behavior. The test suites and native validation must pass separately.

Review the final artifact manually before any publication: verify version alignment, license and legal notes, no vendor DLLs or code objects are unintentionally packaged, no secrets are present, and that documentation makes the supported scope and limitations clear. This repository does not publish, sign or upload artifacts automatically.

## GitHub distribution and project site

The intended initial distribution channel is GitHub: publish the source repository and, when appropriate, attach the checked wheel from `dist/` to a GitHub Release. Do not attach ROCm/HIP vendor DLLs, NVIDIA software, GPU code objects, local build directories, or credentials.

The static project site is in [`docs/site/`](docs/site/). After the project has been committed and pushed, select **GitHub Actions** in the repository's **Settings → Pages** panel. The included Pages workflow deploys this self-contained directory on a qualifying `main` push. See [`docs/site/README.md`](docs/site/README.md) for the exact setup and the boundary of what the workflow does.
