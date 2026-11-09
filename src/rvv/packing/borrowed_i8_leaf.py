"""Opt-in signed-I8/I32 leaves over caller-owned, borrowed buffers.

This authors ordinary MLIR and selects the existing accumulator-resident
feature. It does not pack operands, partition an operation, allocate a team,
or promise a physical issue rate. The caller owns descriptor validation and
must verify the compiler's no-allocation/no-copy postconditions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

INDEX_MAX = (1 << 63) - 1
ROOT = Path(__file__).resolve().parents[3]


class LeafError(ValueError):
    """A requested leaf is outside the explicit source/storage contract."""


def _positive(name: str, value: int, maximum: int = INDEX_MAX) -> None:
    if type(value) is not int or not 0 < value <= maximum:
        raise LeafError(f"{name} must be a positive bounded integer")


def _power_of_two(name: str, value: int, maximum: int) -> None:
    _positive(name, value, maximum)
    if value & (value - 1):
        raise LeafError(f"{name} must be a power of two")


def _divisor(extent: int, limit: int) -> int:
    result = 1
    while result * 2 <= limit and extent % (result * 2) == 0:
        result *= 2
    return result


@dataclass(frozen=True)
class LeafSpec:
    """A static leaf and pitches, independent of any chip or physical DLEN.

    Logical A[M,K] has element strides [1,a_k_stride], B[K,N] has
    [b_k_stride,1], and C[M,N] has [c_row_stride,1]. Descriptor offsets remain
    dynamic. All three borrowed byte spans must be disjoint and CPU-addressable;
    C is naturally I32-aligned. The caller validates offset+span and address
    arithmetic against actual capacities before calling the emitted function.
    """

    m: int
    n: int
    k: int
    a_k_stride: int
    b_k_stride: int
    c_row_stride: int
    vlen_bits: int
    row_block_limit: int = 4
    reduction_chunk_limit: int = 16
    lane_limit: int | None = None

    def __post_init__(self) -> None:
        for name in ("m", "n", "k", "a_k_stride", "b_k_stride", "c_row_stride"):
            _positive(name, getattr(self, name))
        _power_of_two("vlen_bits", self.vlen_bits, 65536)
        if self.vlen_bits < 128:
            raise LeafError("the ordinary RVV contract requires VLEN >= 128")
        _power_of_two("row_block_limit", self.row_block_limit, 4)
        _power_of_two("reduction_chunk_limit", self.reduction_chunk_limit, 16)
        if self.lane_limit is not None:
            _power_of_two("lane_limit", self.lane_limit, self.vlen_bits // 16)
        if self.a_k_stride < self.m:
            raise LeafError("A pitch would overlap logical reduction rows")
        if self.b_k_stride < self.n or self.c_row_stride < self.n:
            raise LeafError("B/C pitch would overlap logical rows")
        if any(size > INDEX_MAX for size in self.minimum_span_bytes.values()):
            raise LeafError("borrowed span exceeds signed RV64 indexing")

    @property
    def minimum_span_bytes(self) -> dict[str, int]:
        """Reachable span from the descriptor's dynamic element offset."""
        return {
            "a": (self.k - 1) * self.a_k_stride + self.m,
            "b": (self.k - 1) * self.b_k_stride + self.n,
            "c": ((self.m - 1) * self.c_row_stride + self.n) * 4,
        }


@dataclass(frozen=True)
class Emission:
    symbol: str
    spec: LeafSpec
    source: str
    mr: int
    nr: int
    kc: int
    feature: str | None

    @property
    def vectorize(self) -> bool:
        return self.feature is not None

    def lowering_kwargs(self) -> dict:
        """Arguments for the existing target-neutral native lowering entry."""
        return {
            "vectorize": self.vectorize,
            # Bufferized one-row tensor slices can retain identity copies and
            # loop-carried ownership flags. Use the existing checked self-copy
            # cleanup; returned LLVM still requires the separate closure gate.
            "features": frozenset({"erase_self_copy"})
            | (frozenset({self.feature}) if self.feature else frozenset()),
            "transform_schedule": (
                (ROOT / "schedules/hand_v0_int8/schedule.mlir").read_text()
                if self.vectorize
                else None
            ),
            "static_arena": False,
        }

    def record(self) -> dict:
        return {
            "schema": "rvv.borrowed_i8_leaf.v1",
            "symbol": self.symbol,
            "c_interface_symbol": "_mlir_ciface_" + self.symbol,
            "spec": asdict(self.spec),
            "block": {"m": self.mr, "n": self.nr, "k": self.kc},
            "feature": self.feature,
            "requested_compiler_cleanup": "erase_self_copy",
            "source_scalar_1x1": not self.vectorize,
            "vector_MAC_instructions_guaranteed": False,
            "numeric_contract": "C += signedI8(A)*signedI8(B), modulo2^32",
            "initial_C": "read; zero seed is a caller policy, not implicit",
            "operand_types": ["i8", "i8", "i32"],
            "memory_space": 0,
            "minimum_span_bytes": self.spec.minimum_span_bytes,
            "required_base_alignment_bytes": {"a": 1, "b": 1, "c": 4},
            "descriptor_offsets": "dynamic, nonnegative and capacity-checked by caller",
            "aliases": "three disjoint borrowed spans; restrict lifetime covers the call",
            "effects": {"a": "read", "b": "read", "c": "read_write"},
            "source_allocation_packing_or_team_creation": False,
            "required_lowering_postconditions": [
                "no heap or memref allocation",
                "no input/output repacking or memrefCopy",
                "destination-passing C ABI",
                "actual machine register/spill audit",
                "full output, input and guard numerical validation",
            ],
            "VLEN_contract": "verify actual vlenb against declared architectural VLEN",
            "DLEN_assumed": False,
            "physical_busy_or_peak_claim": False,
            "limitations": [
                "N=1 may lower to scalar arithmetic with vector storage",
                "1x1 uses explicit ordinary compiler scalar lowering",
                "this does not partition, schedule workers, or implement a model",
            ],
        }


def emit(symbol: str, spec: LeafSpec, *, allow_scalar_1x1: bool = False) -> Emission:
    """Author one ordinary signed MAC with borrowed tensor-SSA accumulation.

    Tensor SSA keeps accumulator values across K; direct memref contraction
    spelling can leave a C load/store inside every reduction iteration. This
    API does not silently scalarize an unsupported multidimensional leaf.
    """
    if not isinstance(spec, LeafSpec):
        raise LeafError("LeafSpec is required")
    if not isinstance(symbol, str) or not symbol.isascii() or not symbol.isidentifier():
        raise LeafError("an ASCII identifier is required for the function symbol")
    if type(allow_scalar_1x1) is not bool:
        raise LeafError("allow_scalar_1x1 must be explicit boolean policy")
    mr = _divisor(spec.m, spec.row_block_limit)
    nr = _divisor(spec.n, spec.lane_limit or spec.vlen_bits // 16)
    kc = _divisor(spec.k, spec.reduction_chunk_limit)
    scalar = spec.m == 1 and spec.n == 1
    if scalar and not allow_scalar_1x1:
        raise LeafError("1x1 requires explicit compiler scalar-tail admission")
    if mr == nr == 1 and not scalar:
        raise LeafError("partition this leaf; a 1x1 register tile is not admitted")
    feature = None
    if not scalar:
        feature = (
            "accumulator_resident_microkernel_v3"
            if (mr, nr, kc) == (4, 16, 16)
            else f"accum_resident_v3_{mr}_{nr}_{kc}"
        )
    m, n, k = spec.m, spec.n, spec.k
    a = f"memref<{m}x{k}xi8,strided<[1,{spec.a_k_stride}],offset:?>>"
    b = f"memref<{k}x{n}xi8,strided<[{spec.b_k_stride},1],offset:?>>"
    c = f"memref<{m}x{n}xi32,strided<[{spec.c_row_stride},1],offset:?>>"
    ta, tb, tc = f"tensor<{m}x{k}xi8>", f"tensor<{k}x{n}xi8>", f"tensor<{m}x{n}xi32>"
    source = f"""module {{
func.func @{symbol}(%a:{a},%b:{b},%c:{c}) attributes {{llvm.emit_c_interface}} {{
 %ta="bufferization.to_tensor"(%a) {{restrict}}:({a})->{ta}
 %tb="bufferization.to_tensor"(%b) {{restrict}}:({b})->{tb}
 %tc="bufferization.to_tensor"(%c) {{restrict,writable}}:({c})->{tc}
 %result=linalg.generic {{indexing_maps=[affine_map<(i,j,k)->(i,k)>,affine_map<(i,j,k)->(k,j)>,affine_map<(i,j,k)->(i,j)>],iterator_types=["parallel","parallel","reduction"]}}
 ins(%ta,%tb:{ta},{tb}) outs(%tc:{tc}) {{
 ^bb0(%x:i8,%y:i8,%acc:i32):
 %xx=arith.extsi %x:i8 to i32
 %yy=arith.extsi %y:i8 to i32
 %product=arith.muli %xx,%yy:i32
 %sum=arith.addi %acc,%product:i32
 linalg.yield %sum:i32
 }}->{tc}
 "bufferization.materialize_in_destination"(%result,%c){{writable}}:({tc},{c})->()
 return
}}
}}
"""
    return Emission(symbol, spec, source, mr, nr, kc, feature)
