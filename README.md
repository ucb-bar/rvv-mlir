# rvv-mlir

Merlin's published codegen packages for the **rvv** target.

This repository is **generated** by Merlin's `merlin-target-publish` bridge. It uses **branch-per-version** publishing, so *this* branch is only a directory — the packages themselves live on the branches below. Check one out to get a standalone, buildable out-of-tree tree plus its provenance under `.merlin/`.

## Published packages

| branch | package | dtype | status | what it is |
|---|---|---|---|---|
| `stable/rvv_tuned_v1_d1_vfmacc_outerproduct` | `rvv_tuned_v1_d1_vfmacc_outerproduct` | `fp32` | `spike_verified` | certified champion |
| `baseline-int8_w8a8` | `hand_v0_int8` | `int8_w8a8` | `spike_verified` | frozen unoptimized control (the before/after reference) |
| `stable/impr_tuned_wholemodel_vf_int8` | `impr_tuned_wholemodel_vf_int8` | `int8_w8a8` | `k1_verified` | certified champion |

## Using a package

```sh
git clone -b <branch> <this-repo> rvv-mlir
cd rvv-mlir
```

An `rvv` package is a **vector schedule**, not a dialect: the payload is a transform-dialect schedule plus the codegen knobs that go with it.

- `payload/schedule.mlir` — the transform-dialect schedule (tiling + vectorization of the contractions)
- `payload/knobs.yaml` — `cflags`, `dtype_strategy`, `op_match` tile/vector sizes, `lmul_policy`, and the `expected_instructions` the emitted code must contain
- `payload/baseline_runs/` — the recorded reference runs

Merlin consumes it through `merlin.rvvgen.registry.load_rvv_package(<dir>)` and applies it with `merlin.rvvgen.apply.apply_rvv_package(...)`; the schedule and cflags are the only things that change, so the rest of the pipeline is untouched.

The `baseline` branch is the FROZEN, hand-authored, unoptimized control. It exists so a speedup claim can be reproduced against the same before/after this repo published, not against a moving target.

## Provenance

Each commit on a package branch is one promotion, and its message embeds the champion package id, the internal run id, the Merlin git sha and the certification summary. History is the provenance trail; the branch tip is the current champion.

Generated from Merlin `5d68ab4`.
