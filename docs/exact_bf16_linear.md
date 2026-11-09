# Exact-order BF16 linear leaf

`backend.exact_bf16_linear_api()` authors an ordinary MLIR leaf for a closed,
static linear operation. Selection is explicit and off by default:

```python
api = backend.exact_bf16_linear_api()
spec = api.LinearSpec(m=1, k=2048, n=32000)
report = api.compile_kernel(
    spec, work=new_private_directory, compiler=rv64_c_compiler,
    mlir_opt=mlir_opt, mlir_translate=mlir_translate, clang=rv64_clang,
    enabled=True, march="rv64gcv",
)
```

Select RISC-V-capable `clang` or `llc` for LLVM code generation. Chip packages
may override `march`; scheduling priority belongs to the caller. The region ABI
is `int rvv_exact_bf16_linear(void *, const void *const *, void *const *)`.
Custom symbols use the `rvv_exact_bf16_linear_` namespace. Inputs are contiguous
BF16 A[M,K], original BF16 W[N,K], and positive-zero F32 seed[M,N]; output is
BF16 [M,N]. Naturally aligned buffers and valid capacities are caller obligations.

The leaf uses ascending-K SCF iteration, integer widening, unflagged
`arith.mulf`/`arith.addf`, and one integer BF16 RNE conversion. Shared LLVM
vector/finalization passes bypass tuned contraction and reassociation schedules.
Clang uses `-ffp-contract=off`; llc uses `-fp-contract=off`. LLVM IR and final
proof objects are independently audited for fast math or fused arithmetic.

Pass `target=api.PanelTarget(...)` to select VLEN, operand SEW16, accumulator
SEW32, LMUL, register budget and workspace through the existing panel policy.
Lanes are bounded by VLEN times LMUL divided by 16, N and workspace capacity;
row blocks by the register budget (four by default). Full blocks share masked
original-NK gathers and widening across independent row accumulators. Remaining
rows use a single-row loop; masked N tails preserve bounds. The default target
uses VLEN128 and LMUL1. Larger targets require a matching minimum-VLEN suffix
such as `rv64gcv_zvl256b` and execution hardware satisfying that minimum.

Portable C admission checks finite inputs, zero seed, RNE and output disjointness
using `fegetround` (link the C floating-point environment, e.g. `-lm`). Private
output storage leaves output unchanged on failure: -1 for invalid pointers,
overlap or rounding, -2 for numeric domain failure, -3 for allocation failure.
Selection adds no arbitrary linalg matcher, Merlin CLI route or placement change.

The reference is `c_bf16_linear` on `reed-nicolas/merlin`'s `feat/tinyllama` at
[`49673bc57223f2da042bca01bee26d31d8ca8829`](https://github.com/reed-nicolas/merlin/commit/49673bc57223f2da042bca01bee26d31d8ca8829),
associated with https://github.com/ucb-bar/merlin/pull/63; it differs from the
compiler API pin on `feat/OPU-VME`. The harness transposes W[N,K] bit-for-bit
for the reference's B[K,N]. The proof uses shipped arithmetic flags and covers
VLEN128/256/512, tails, signed zeros, subnormals, overflow and refusals. Spike
retired instructions measure instruction work, not hardware cycles.
