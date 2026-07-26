# rvv-mlir

Standalone, buildable out-of-tree Merlin codegen backend for **rvv** (family `vector_schedule`).

This repository is **generated** by Merlin's `merlin-target-publish` bridge: it is the certified champion codegen package for the target, exported as its own repo. The buildable tree at the repo root *is* the content; the package manifest + provenance ride along under `.merlin/`.

## What

- Champion package: `impr_tuned_wholemodel_vf_int8`
- Family: `vector_schedule`
- Recorded status: `k1_verified`
- Merlin git sha (this export): `5d68ab4`

## How to build

```sh
git clone <this-repo> rvv-mlir
cd rvv-mlir
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bin/rvv-opt --version
```

The codegen payload (schedule/knobs for rvv; dialect/lowering/contracts for gemmini) lives under `payload/`.

## Provenance

- Certification: `pass`
- Certified by run: `k1_int8_wholemodel_ab_tiny_llama`
- Fingerprint: `2255a08aa53d9c66850bb9d0412a8ae9988e2211389221dfcc6565886b7a3d22`

See `.merlin/provenance.yaml` and `.merlin/certification.yaml` for the full lineage. Each commit on this repo is one promotion; the history is the provenance trail.
