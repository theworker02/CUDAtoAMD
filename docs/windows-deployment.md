# Windows Deployment Design

The Windows ABI target is `compatcuda.dll`, built from the independently implemented header surface. Compatibility deployment must be explicit: an application must be built against these headers and import library, or launched with an owner-configured compatibility environment. The project will not use DLL injection or stealth interposition.

The current build is an ABI smoke target only. It requires no ROCm runtime and returns `cudaErrorNotSupported`. A functional HIP-enabled Windows build must validate the compatible HIP runtime, compiler path, AMD driver, target architecture, and every enabled API family before launch.
