# Borrowed signed-I8/I32 leaves

`backend.borrowed_i8_leaf_api()` is an opt-in source adapter for caller-owned
buffers. `emit(symbol, LeafSpec(...))` returns ordinary MLIR, an existing
accumulator-resident feature selection, and an explicit ABI/effect record.
It creates no worker team, input pack, output copy or model dispatcher.

Logical A[M,K] uses strides `[1,a_k_stride]`, B[K,N] uses
`[b_k_stride,1]`, and C[M,N] uses `[c_row_stride,1]`. The emitted C interface
takes three ranked two-dimensional MLIR memref descriptors. Their offsets
remain dynamic. The caller must validate nonnegative offsets, actual byte
capacities, signed address arithmetic, CPU accessibility, disjoint borrowed
spans, and four-byte C alignment before execution. The record supplies minimum
reachable spans; it does not validate actual pointers.

The numerical operation is `C += signed_i8(A) * signed_i8(B)` with I32
accumulation modulo 2^32. C is read as the initial seed. A caller that needs a
zero seed must initialize it explicitly. No unsigned conversion, bias, scale,
quantization or floating-point effect is inserted.

Borrowed tensor SSA carries accumulators across K and materializes the result
in the supplied C descriptor. The caller runs the existing native lowering
with `emission.lowering_kwargs()`, then checks the required postconditions:
no allocator or repacking/copy, destination-passing ABI, machine register/spill
audit, and complete output/input/guard validation. Source authorship is not a
substitute for those compiler-output checks.

Exact power-of-two register blocks are bounded by declared architectural VLEN
and widening source SEW16. Physical DLEN is neither an input nor inferred from
VLEN. The caller must check actual `vlenb` and observe the resulting VL/SEW/LMUL
when reporting lane coverage. The selected feature does not guarantee a
particular machine instruction, physical busy fraction or peak issue rate.

Partitioning belongs to the caller. A leaf whose exact register tile would be
1x1 is refused unless the entire leaf is 1x1 and scalar lowering was explicitly
admitted. N=1 with multiple rows can compile to scalar arithmetic with vector
storage; it is not advertised as vector MAC arithmetic. These conservative
tails retain the original numerical operation without an authored C kernel.
