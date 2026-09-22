# Security

**Date:** 2026-09-21

## Reporting

Contact GitHub [@theworker02](https://github.com/theworker02) privately for vulnerabilities. Do not file public issues with exploit details.

## Trust model

| Surface | Posture |
| --- | --- |
| `compat analyze` / `port-plan` | Read-only; treats project trees as **untrusted input**; does not execute analyzed project code; skips `.git` |
| Native HIP runtime | Loads operator-built libraries and trusted `.hsaco` modules the user supplies |
| `compat run` / `compat python` | Launches operator-chosen programs in an explicit environment — not a sandbox claim |
| Wheel preflight | Heuristic blockers only; not a malware scanner |
| Remote execution | **Not implemented**; any future remote path must be explicit opt-in |

## Guidance for translators / compilers

Generated kernels, binary inputs (PTX, fatbins), caches, and any future remote protocol messages must be treated as untrusted. Prefer fail-closed errors over silent partial emulation.

## Acquisition note

No default telemetry. Secrets (if any) are operator-local ROCm paths and optional library paths — rotate nothing Hub-side because this project is not a hosted SaaS.
