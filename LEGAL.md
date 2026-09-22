# Legal Notes

**Date:** 2026-09-21  
**Status:** Diligence aid. Not legal advice.

## Independence

CUDAtoAMD is **independent interoperability infrastructure**. It must not:

- Redistribute proprietary NVIDIA binaries, libraries, or headers copied from NVIDIA SDKs
- Copy proprietary NVIDIA source into this tree
- Imply NVIDIA or AMD endorsement or certification
- Represent AMD hardware as NVIDIA hardware, or spoof NVIDIA tooling (`nvidia-smi`, fake compute capability)

## License

- Current repository: **source-available proprietary** — see [`LICENSE`](LICENSE) and [`COMMERCIAL.md`](COMMERCIAL.md)
- Transition letter: [`LICENSE_TRANSITION_NOTICE.md`](LICENSE_TRANSITION_NOTICE.md)
- Acquisition brief: [`ACQUISITION.md`](ACQUISITION.md)

## Third-party SDKs at runtime

HIP / ROCm, hipBLAS, hipFFT, and related AMD tooling remain under **their** licenses and EULAs. Sellers do not transfer AMD’s or NVIDIA’s IP. Buyers must obtain and comply with AMD ROCm/HIP terms on their machines.

## Clean-room posture

Compatibility headers and adapters in this repository are intended as **original** work implementing documented or observed interfaces, not as a dump of CUDA source. New API surfaces and dependencies require license review before merge.

## Acquisition packaging

What typically transfers vs what does not is summarized in [`docs/acquisition/`](docs/acquisition/) and [`COMMERCIAL.md`](COMMERCIAL.md). Trademark registration status for the CUDAtoAMD mark is **UNKNOWN** unless counsel confirms otherwise.
