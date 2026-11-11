# Exact byte transpose

`backend.exact_code_transpose_api()` exposes a default-off byte transpose from `[groups,rows,columns]` to `[groups,columns,rows]`. `CodeTransposeSpec(groups, rows, columns, lanes=16, symbol="rvv_code_transpose")` accepts positive static geometry and power-of-two vector widths up to 64, checks signed-index spans, and derives every loop bound, gather stride, mask, and ABI descriptor. Custom symbols use the `rvv_code_transpose_` namespace. `emit_mlir(spec, enabled=True)` and `emit_wrapper(spec, enabled=True)` preserve payload bits, add no allocation, and reject null, overlapping, or wrapping buffers before writes.

The MLIR uses `vector.gather` and masked stores and lowers through the exact BF16 leaf's `PIPELINE`. For the measured VLEN-256 target, LLVM selects RVV strided byte loads and masked contiguous byte stores. There is no arithmetic conversion, floating-point requirement, special BF16 vector instruction, or automatic graph routing.

The native tests cover independent shapes, exact and partial vectors, multiple groups, unaligned pointers, all byte values, guards, input preservation, invalid geometry, alias rejection, and address wrapping. They do not measure hardware timing.
