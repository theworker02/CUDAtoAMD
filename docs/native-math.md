# Native math subset (0.12.0)

These are explicit AMD API adapters, not cuBLAS/cuFFT replacement DLLs. The C headers are amd_blas.h and amd_fft.h; Python exposes Blas and FFTPlan from compat.native.

## GEMM

amdBlasCreateOnDevice selects a device. amdBlasGemmStridedBatchedEx accepts a stream, F32/F16/BF16 input storage, FP32 alpha/beta, FP32 accumulation and FP32 output. Matrices are column-major. Both transposes and nontrivial leading dimensions are supported.

Strides are in elements of each matrix's storage type. Zero A/B stride broadcasts an input across batches; C batches must not overlap. Invalid enum values, negative sizes, too-small leading dimensions, short batch strides and stride overflow are rejected. Raw C pointers must refer to sufficiently large live device allocations. Python additionally checks buffer extents and disallows output aliasing inputs.

Python use: Blas(device).gemm(a,b,c,m=...,n=...,k=...,input_type="bf16",batches=...,stride_a=...,stride_b=...,stride_c=...). Use a context manager around Blas. Input BF16 buffers contain raw little-endian 16-bit encodings; no implicit precision conversion is performed. All Python buffers must share their allocation stream.

Verified on gfx1101: F32/F16/BF16, all four transpose combinations, two batches, padded layouts, alpha=1.25 and beta=0.5. This does not establish every size, corner case, exceptional floating-point value or hardware target. No performance or zero-overhead claim is made.

## FFT

FFTPlan(device,length,batches=1) implements contiguous batched 1D complex FP32 transforms. Complex values are interleaved real/imaginary floats (AmdComplex32 in C). execute(source,destination) is forward; inverse=True is inverse. Identical input/output buffers enable in-place execution. Inverse is unnormalized: divide results by length for a round trip.

hipFFT is optional at build time. With no hipFFT or COMPATCUDA_ENABLE_FFT=OFF, FFT functions remain exported and return AMD_ERROR_NOT_INITIALIZED (backend unavailable). The Python API reports that native status rather than claiming success.

Verified against a CPU complex DFT for lengths 7 and 8, two batches, and in-place inverse round trips. Real transforms, FP64, multidimensional plans and arbitrary strides are not implemented.

Plan/BLAS destruction synchronizes the associated device. Do not race handle operations or destruction; keep streams and buffers alive until work finishes. HIP-enabled distributions still need matching vendor DLLs installed separately.
