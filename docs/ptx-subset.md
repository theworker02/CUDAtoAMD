# Experimental PTX compilation (0.27.0)

`compat ptx-object` explicitly translates a small, validated PTX subset to HIP source, then uses HIP-Clang to produce native AMD ELF. It is not an LLVM PTX backend, a runtime JIT, or a way to run arbitrary CUDA wheels. No NVIDIA binary or implementation is bundled.

From the repository root, with the HIP SDK installed:

```powershell
python -m compat ptx-object examples/vector_add.ptx -o vector-from-ptx.hsaco --arch gfx1101
python -m unittest discover -s tests -p test_ptx.py -v
```

The output must be a new filename. The example's exported kernel is `ptx_vector_add`, taking three 64-bit device addresses in A/B/output order. Load the resulting code object with the existing native Module or Driver module API. The fixture has **no bounds guard**: allocate all three arrays for exactly `grid.x * block.x` float elements, with other dimensions equal to one. The test demonstrates allocations, transfers, launch and complete output comparison. Code objects and PTX inputs must be trusted; validation is not memory isolation.

## Accepted grammar

- Exactly one `.visible .entry` with `.version 7.0`, `.target sm_50`, `.address_size 64`, in that order. The sm target is input metadata, not a claim about AMD hardware; `--arch` selects the real AMD target.
- Scalar `.param` and `.reg` types: `.u32`, `.u64`, `.f32`. Register arrays use `%name<N>`. Parameters and register declarations are typed and duplicates rejected.
- `ld.param.u32/u64/f32`, `mov.u32/u64/f32`, add/subtract, low and wide integer multiply, u32 bitwise operations, u32/u64 and u32-to-FP32 conversions, `add.rn.f32`, `mul.rn.f32`, `fma.rn.f32`, u32/f32 global loads/stores, and terminal `ret`.
- `.pred` registers, unsigned less/equal/not-equal/greater comparisons, labels and forward `bra` / predicated `@%p` or `@!%p bra`. Backward branches and loops are rejected deliberately.
- Thread/block special registers `%tid`, `%ctaid`, `%ntid`, `%nctaid` with x/y/z suffixes for u32 operands. Global addresses must be initialized u64 registers without inline offsets. Integer immediates are unsigned decimal; floating literals are not accepted.
- Straight-line code only. Registers must be initialized before use. Only `//` comments are supported. Input is limited to 1 MiB, 64 parameters and 1024 registers.

This grammar follows selected documented [PTX instruction contracts](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html). Integer operations use unsigned-width arithmetic; explicit float addition lowers to HIP's round-to-nearest addition intrinsic. Passing the current tests does not establish exhaustive floating-point edge-case or memory-model conformance.

## Diagnostics and boundaries

Errors use `PTX-E001` through `PTX-E006` for size, module grammar, parameters, statement structure, operands and unsupported instructions. The entire input is checked before compiler execution, with no silent instruction dropping or opcode substitution. Compiler failures preserve the user's source and do not publish an output artifact.

Signed register types, functions/calls, atomics, barriers, shared memory, vector loads, tensor/warp instructions, debug directives, broader target/version grammar, NVVM and cubins remain unsupported. Real framework-generated PTX will normally require many of these features. Transparent PyTorch/Triton CUDA execution is not implemented.

Verified on Windows HIP SDK 7.1 and RX 7800 XT gfx1101. Tests execute float vector addition with one and three blocks, a bounds-safe 179-element launch using 192 threads, a 32-bit affine transform with wide address multiplication, and a bounds-safe 129-element FP32 SAXPY using FMA with 192 threads. Invalid syntax/types/uninitialized registers are rejected. Other targets and general instruction semantics remain unverified.
