# rvv-mlir

Standalone, buildable out-of-tree Merlin codegen backend for **rvv** (family `vector_schedule`).

This repository is **generated** by Merlin's `merlin-target-publish` bridge: it is the certified champion codegen package for the target, exported as its own repo. The buildable tree at the repo root *is* the content; the package manifest + provenance ride along under `.merlin/`.

## What

- Champion package: `sol_k1_int8`
- Family: `vector_schedule`
- Recorded status: `k1_verified`
- Merlin git sha (this export): `abe6cb3`

## How to build

```sh
git clone <this-repo> rvv-mlir
cd rvv-mlir
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bin/rvv-opt --version
```

## Provenance

- Certification: `recorded:k1_verified`
- Certified by run: `n/a`
- Certified against: **tier not recorded** — do not read this as an RTL result
- Fingerprint: `n/a`

See `.merlin/provenance.yaml` and `.merlin/certification.yaml` for the full lineage. Each commit on this repo is one promotion; the history is the provenance trail.
