# Architecture

```text
CUDA-oriented source → analyzer → compatibility database → report / HIPIFY handoff
                                             │
                                    future dispatcher
                                             │
                                HIP adapter → ROCm → AMD GPU
```

The analyzer is deliberately separate from execution code so it can be useful in CI without ROCm installed. The repository now also contains a separately named `compatcuda` ABI target and project-owned CUDA-facing declaration subset. With no HIP backend enabled, the library returns explicit `cudaErrorNotSupported` results rather than emulating successful GPU work.

The database is the source of truth. `compat matrix` renders `COMPATIBILITY.md`; tests validate the schema. A future runtime must consume the same IDs for diagnostic messages and conformance results.
