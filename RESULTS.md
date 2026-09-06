# SOL — Speed of Light for int8 (W8A8) whole-model codegen on the SpaceMiT K1

**SOL is the ceiling, not a claim.** This package records what the hardware *can* do, what
ExecuTorch and merlin actually do against it, and the mechanisms that account for the difference.
It exists so that analysis found by hand becomes a **target the compiler can be measured against**
rather than knowledge that evaporates.

**This configuration does not beat ExecuTorch on any model in the study.** Every number below is
measured; nothing here is a projection.

## The roof

Compute roof, int8: **51.2 GMAC/s** = 32 MACs per `vwmacc` (VLEN=256, SEW=8→16 widening) at 1.6 GHz.

| model | MACs/inference | compute SOL | ExecuTorch | merlin | ET/SOL | ours/SOL | ours/ET |
|---|---|---|---|---|---|---|---|
| lstmnetvit | 0.056 G | 1.1 ms | 49.4 ms | 89.0 ms | 45.3× | 81.6× | **1.80×** |
| resnet50_v1_5 | 4.124 G | 80.5 ms | 836.3 ms | 2814.6 ms | 10.4× | 34.9× | **3.37×** |
| tiny_llama | 8.285 G | 161.8 ms | 1137.1 ms | 22462.7 ms | 7.0× | 138.8× | **19.76×** |

**Read the `ET/SOL` column before optimizing anything.** ExecuTorch is itself 7–45× off the compute
roof. Beating it does not require reaching SOL — only closing more of the same gap than it has.

### The memory roof, which is what actually binds

Compulsory traffic is the B-operand bytes the contractions need. *Touched* is 64-byte lines actually
pulled, derived from the emitted loop's stride.

| model | compulsory | touched | amplification | ET pace | our pace |
|---|---|---|---|---|---|
| lstmnetvit | 5.9 MB | 73.6 MB | **12.5×** | 0.12 GB/s | 0.83 GB/s |
| tiny_llama | 1035 MB | 8284 MB | **8.0×** | 0.91 GB/s | 0.37 GB/s |

`tiny_llama` is a **pure bandwidth** problem: 1.3 GB of weights read once per inference, ExecuTorch
moving them at ~1.15 GB/s. Its compute SOL is 162 ms and it takes 1137 ms — arithmetic is irrelevant.

> **UNMEASURED, and the most valuable missing number:** the K1's actual single-core DRAM bandwidth.
> Every "pace" above is an *achieved* rate, not a roof. Until a STREAM-style measurement exists the
> memory roofline has no ceiling term, and it is not known how much of ExecuTorch's 1.15 GB/s is the
> hardware and how much is ExecuTorch. **Measure this first.**

## Why we are off the roof

1. **The weight stream is unpacked.** The micro-kernel loads 16 contiguous bytes of B, then advances
   by N. One fresh 64-byte line per k-step with **16 of 64 bytes consumed**, re-touched at a reuse
   distance of K lines (128 KB at K=2048). ExecuTorch pays a one-time `load_ns` (**58.5 s** on
   tiny_llama, 1.16 s on resnet50, 0.17 s on lstmnetvit) to prepack weights into a layout its GEMM
   reads contiguously — a cost *outside* the execute line every ratio here divides by.
2. **The MAC runs at 12.5% of the achievable vector width.** SEW/LMUL in force at each `vwmacc`:
   `e16,m1` 89.5%, `e16,mf2` 10.5% — 16 and 8 elements against the 128 of an `e16,m8` group.
   `op_match` pins `vector: [4, 8, 1]`.
3. **The operand feed is scalar.** Per `vwmacc`: exactly 1.00 `lbu` and 1.00 `vmv.v.x` — B fetched a
   byte at a time through the scalar unit and broadcast. 11.08 instructions per MAC.

**The inner loop is NOT the defect.** `blk_mm_4x16` is 14 instructions per k-step for 4 `vwmacc` over
64 MAC lanes; accumulators are register-resident with zero vector stores in the loop; there is no
`@memrefCopy` in `forward`; address arithmetic is 14.3%. Static issue floor: **5–10% of measured
cycles**. Effort spent on the loop body is misdirected.

## Refuted — do not re-try without new evidence

| lever | result |
|---|---|
| 4-lever "epilogue" stack (`vectorize_non_contraction_generics`, `fuse_quantize_round_convert`, `perop_mr_fill_register`, `im2col_panel_pack`) | **REGRESSION** on the board: lstmnetvit 1.802× → 1.863×; `forward` 42,124 → 49,868 instructions (linked ELF); numerics bit-identical |
| direct-conv arm (`conv_register_block`) on resnet50 | unreachable: im2col is 0.51% of work; all 17 expanding convs are padded and `_try_direct_conv2d` refuses padding, so 0/17 divert; every diverted build fails to link on `aten_convolution_*` |
| `vectorize_amax_reduction` | +1,641 instructions; clang already vectorized that reduction (`vfredmax.vs` 54 → 1) |

Recurring mechanism in two of the three: **a fixed-width tile displaces clang's own vectorization.**
69.0% of the vector instructions in resnet50's `forward` are clang's, not this schedule's.

## Open, ranked by expected value

1. **Weight panel packing** — pack B into NR-wide panels at build time so a 64-byte line carries 4
   useful k-steps instead of 1. Directly attacks the 8–12.5× amplification. Not yet built.
2. **Width** — `perop_nr_fill_register`, `lmul_group_m2/m4`, wider `accum_resident_v3_*` tiles.
3. **`accumulator_resident_wholemodel_vf`** — the PRIOR published champion's only feature
   (`stable/impr_tuned_wholemodel_vf_int8`, `k1_verified`). It keeps **NR=32** with accumulator
   residency and A-scalarizes so the K-loop emits a scalar-operand MAC instead of a ~20-instruction
   broadcast ladder (~3 ops/FMA instead of ~20). **This package omits it, and our derived block is
   `blk_mm_4x16` = NR=16, half that width.** Whether we regressed against the prior champion is
   queued and UNKNOWN.
4. **Requant fusion into the contraction tile** — merlin `b8f5abec`. Removes a full traversal of
   every contraction output (2.97 MB lstmnetvit / 44.46 MB resnet50). Static: −5.3% instructions,
   vector +1.6%. Unmeasured on the board.
5. **Honour `pkg.cflags`** — they never reach the model object today; honouring them removes 5,771
   strided `vlse32`/`vsse32` that clang introduced.

## Accuracy

Both arms pass the derived bar (`baselines.bundle.int8_accuracy_bar`: `rel = max(ABSOLUTE_INT8_REL,
QUANT_EXCESS_K × floor_rel)`) that the ExecuTorch arm is judged by:

| model | shared bar | merlin | ExecuTorch |
|---|---|---|---|
| lstmnetvit | cos>0.99, rel<0.05 | 0.99998 / 0.0094 ✅ | 0.99999 / 0.0039 ✅ |
| resnet50_v1_5 | cos>0.99, rel<0.05 | 0.99828 / 0.0472 ✅ | 0.99959 / 0.0305 ✅ |
| tiny_llama | cos>0.9487, rel<3.8325 | 0.97792 / 0.1769 ✅ | 0.99434 / 0.1063 ✅ |

merlin's own `_gate` T1 (`cos>0.999, rel<1e-2`) is **absolute** and refuses resnet50 and tiny_llama.
That is a stricter, different question (did we implement W8A8 as the reference does) and should stay
tight — but the two arms are not judged by the same rule, and on tiny_llama the bars differ ~380× in
`rel`. `resnet50_v1_5`'s `golden.npy` is byte-identical to its `golden_w8a8.npy`, so it has **no
independent fp32 reference** and its quantization floor is UNKNOWN, not zero.

## Instrument defects found while producing these numbers

- **`codegen_census` reads the unrelocated object**, so its `entry_instructions` understates the
  linked ELF — tiny_llama reports 2,063 where the linked `forward` is 252,754 (**122×**). The gate is
  sound (it sums the whole object); the reported field is not. Always measure the LINKED ELF.
- **`pkg.cflags` never reach the compiler** on the K1 path, so every static vectorization claim taken
  through it measures merlin and clang together.
- An objdump census read the **operand** field instead of the mnemonic (`addr: enc \t MNEMONIC \t
  operands` — mnemonic is tab-field 1). Fixed in merlin `c53cde51`; published figures that carried it
  were corrected there.

## Reproducing

```sh
merlin_commit=abe6cb36
.venv/bin/python build_tools/scripts/k1_int8_fair_compare.py \
  --model lstmnetvit --model-dir out/artifacts/recaptures/lstmnetvit_int8_consistent \
  --baseline out/artifacts/targets/rvv/sol_k1_int8 \
  --ref-cpu-threads 1 --n 3 --warmup 2 --iters 5 --et-n-lo 1 --et-n-hi 4
```

Protocol: min-of-3 launches, each min-of-5 timed after 2 warmup; ExecuTorch `pt2e_qd8` interleaved
**inside** each cell at N=1 and N=4 so its reference is a warm slope, not a cold shot; both arms
matched at **one core**; unpinned (`taskset -c 0` costs ~5× — cpu0 is the interrupt core).
Board noise band 2.6%; a delta inside it is not a result.

`prepack_weight_layout` **refuses on resnet50** (no hoistable weight transposes) and fails the cell
closed — omit it for that model.
