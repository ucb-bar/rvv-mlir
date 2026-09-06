# rvv-mlir

Merlin's published codegen packages for the **rvv** target.

This repository is **generated** by Merlin's `merlin-target-publish` bridge. It uses **branch-per-version** publishing, so *this* branch is only a directory — the packages themselves live on the branches below. Check one out to get a standalone, buildable out-of-tree tree plus its provenance under `.merlin/`.

## Published packages

| branch | package | dtype | status | what it is |
|---|---|---|---|---|
| `stable/rvv_tuned_v1_d1_vfmacc_outerproduct` | `rvv_tuned_v1_d1_vfmacc_outerproduct` | `fp32` | `spike_verified` | certified champion |
| `baseline-int8_w8a8` | `hand_v0_int8` | `int8_w8a8` | `spike_verified` | frozen unoptimized control (the before/after reference) |
| `stable/impr_tuned_wholemodel_vf_int8` | `impr_tuned_wholemodel_vf_int8` | `int8_w8a8` | `k1_verified` | certified champion |
| `stable/sol_k1_int8` | `sol_k1_int8` | `int8_w8a8` | `k1_verified` | certified champion |

## Using a package

```sh
git clone -b <branch> <this-repo> rvv-mlir
cd rvv-mlir
```

An `rvv` package is a **vector schedule**, not a dialect: the payload is a transform-dialect schedule plus the codegen knobs that go with it.

- `payload/schedule.mlir` — the transform-dialect schedule (tiling + vectorization of the contractions)
- `payload/knobs.yaml` — `cflags`, `dtype_strategy`, `op_match` tile/vector sizes, `lmul_policy`, and the `expected_instructions` the emitted code must contain
- `payload/baseline_runs/` — the recorded reference runs

Merlin consumes it through `merlin.mining.registry.load_rvv_package(<dir>)` and applies it with `merlin.mining.apply.apply_rvv_package(...)`; the schedule and cflags are the only things that change, so the rest of the pipeline is untouched.

The `baseline` branch is the FROZEN, hand-authored, unoptimized control. It exists so a speedup claim can be reproduced against the same before/after this repo published, not against a moving target.

## Compiling a model with it

This repository is the **backend**: the target's codegen payload plus its capability contract. The thing that compiles a model is Merlin, which consumes this repo. You need both, and the loop is three commands.

```sh
# 1. Merlin itself (the driver, the frontend, the runtime)
git clone https://github.com/ucb-bar/merlin.git && cd merlin
cp .env.example .env          # then point MERLIN_* at your toolchain / simulators

# 2. Fetch THIS repo as the target's out-of-tree backend
merlin-target-fetch rvv --champion <branch from the table above>

# 3. Compile a workload onto it
merlin-compile --workload <workload> --target rvv --verify
```

`merlin-target-fetch` clones the chosen branch into `out/build/generated/rvv/`, and the target registry then resolves the capability contract and this codegen payload together — so which champion you compile against is the branch you fetched, recorded rather than implied.

`merlin-compile` takes `--run {none,host,spike,verilator,zephyr,k1}` and `--verify`. Start with `--run host` to check the lowering is numerically right, then move up the oracle ladder; `--verify` gates the answer against the workload's golden rather than reporting that something merely ran.

**What you need beyond this repo**: an LLVM/MLIR install matching the `llvm:` block of the package manifest (the out-of-tree C++ API moves between versions), a RISC-V toolchain, and whichever simulator your chosen `--run` needs. Merlin's `docs/guides/getting_started.md` is the base install; `docs/guides/adding_a_target.md` explains the contract this repo carries.

## Provenance

Each commit on a package branch is one promotion, and its message embeds the champion package id, the internal run id, the Merlin git sha and the certification summary. History is the provenance trail; the branch tip is the current champion.

Generated from Merlin `abe6cb3`.
