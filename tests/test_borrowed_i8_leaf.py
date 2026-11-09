"""Borrowed leaves expose explicit types, effects, pitches and refusal bounds."""

import json

import pytest
from merlin.common import mlir_query as mq
from test_backend import BACKEND

API = BACKEND.borrowed_i8_leaf_api()


def spec(**changes):
    values = {
        "m": 2,
        "n": 64,
        "k": 17,
        "a_k_stride": 3,
        "b_k_stride": 65,
        "c_row_stride": 65,
        "vlen_bits": 256,
    }
    return API.LeafSpec(**{**values, **changes})


@pytest.mark.parametrize("vlen,nr", [(128, 8), (256, 16), (512, 32), (1024, 64)])
def test_exact_register_block_uses_architectural_vlen(vlen, nr):
    leaf = API.emit("leaf", spec(vlen_bits=vlen))
    assert (leaf.mr, leaf.nr, leaf.kc) == (2, nr, 1)
    assert leaf.feature == f"accum_resident_v3_2_{nr}_1"
    assert not leaf.record()["DLEN_assumed"]


def test_source_is_borrowed_tensor_ssa_with_explicit_destination():
    leaf = API.emit("leaf", spec())
    module = mq.parse(leaf.source)
    # xDSL's current ToTensorOp constraint reconstructs a layoutless memref;
    # native MLIR verification in prove_borrowed_i8_leaf.py is authoritative
    # for borrowed strided memrefs. This test checks the parsed source structure.
    assert len(list(mq.walk(module, "bufferization.to_tensor"))) == 3
    assert len(list(mq.walk(module, "bufferization.materialize_in_destination"))) == 1
    assert len(list(mq.walk(module, "arith.extsi"))) == 2
    assert len(list(mq.walk(module, "arith.muli"))) == 1
    assert len(list(mq.walk(module, "arith.addi"))) == 1
    assert not list(mq.walk(module, "linalg.fill"))
    assert not list(mq.walk(module, "memref.alloc"))
    assert "strided<[1,3],offset:?>" in leaf.source
    assert "strided<[65,1],offset:?>" in leaf.source


def test_effects_spans_and_lowering_obligations_are_not_hidden():
    record = API.emit("leaf", spec()).record()
    assert record["minimum_span_bytes"] == {"a": 50, "b": 1104, "c": 516}
    assert record["effects"] == {"a": "read", "b": "read", "c": "read_write"}
    assert record["required_base_alignment_bytes"] == {"a": 1, "b": 1, "c": 4}
    assert record["initial_C"].startswith("read")
    assert record["required_lowering_postconditions"]
    assert record["vector_MAC_instructions_guaranteed"] is False
    json.dumps(record)


def test_single_cell_requires_explicit_compiler_scalar_admission():
    scalar = spec(m=1, n=1)
    with pytest.raises(API.LeafError, match="explicit compiler scalar"):
        API.emit("leaf", scalar)
    leaf = API.emit("leaf", scalar, allow_scalar_1x1=True)
    assert leaf.feature is None
    assert leaf.lowering_kwargs() == {
        "vectorize": False,
        "features": frozenset(),
        "transform_schedule": None,
        "static_arena": False,
    }


def test_unsupported_odd_leaf_refuses_instead_of_silent_scalarization():
    with pytest.raises(API.LeafError, match="partition"):
        API.emit("leaf", spec(m=3, n=65))


def test_column_tail_is_compiler_owned_and_does_not_promise_vector_macs():
    leaf = API.emit("leaf", spec(n=1))
    assert leaf.feature == "accum_resident_v3_2_1_1"
    assert leaf.record()["vector_MAC_instructions_guaranteed"] is False


@pytest.mark.parametrize("symbol", ["", "a-b", "x\nreturn", "å", "1leaf", 'x"y'])
def test_symbol_cannot_inject_or_change_ir(symbol):
    with pytest.raises(API.LeafError):
        API.emit(symbol, spec())


@pytest.mark.parametrize(
    "changes",
    [
        {"m": 0},
        {"n": -1},
        {"k": True},
        {"a_k_stride": 1},
        {"b_k_stride": 63},
        {"c_row_stride": 0},
        {"a_k_stride": -1},
        {"vlen_bits": 64},
        {"vlen_bits": 192},
        {"vlen_bits": 131072},
        {"row_block_limit": 3},
        {"row_block_limit": 8},
        {"reduction_chunk_limit": 32},
        {"lane_limit": 32},
        {"lane_limit": 0},
        {"a_k_stride": (1 << 63) - 1},
        {"c_row_stride": (1 << 62)},
    ],
)
def test_invalid_index_pitch_and_vector_budgets_fail_closed(changes):
    with pytest.raises(API.LeafError):
        spec(**changes)


def test_explicit_n_lane_limit_can_only_reduce_the_architectural_budget():
    leaf = API.emit("leaf", spec(lane_limit=8))
    assert leaf.nr == 8
    assert "accum_resident_v3_2_8_1" in leaf.lowering_kwargs()["features"]


def test_default_four_by_sixteen_feature_retains_existing_name():
    leaf = API.emit("leaf", spec(m=4, k=16, a_k_stride=4))
    assert leaf.feature == "accumulator_resident_microkernel_v3"
