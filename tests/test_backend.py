"""Executable RVV provider: pass ordering and published schedules."""

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rvv_provider_backend", ROOT / "backend.py"
)
assert SPEC is not None and SPEC.loader is not None
BACKEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKEND)


def test_backend_pipeline_matches_frozen_compiler_pass_order(monkeypatch):
    for key in (
        "MERLIN_FUSE_POST",
        "MERLIN_PAR_RESIDUE_PARALLEL",
        "MERLIN_NO_DEALLOC",
        "MERLIN_GENERALIZE_BEFORE_FUSE",
    ):
        monkeypatch.delenv(key, raising=False)
    assert BACKEND.API_VERSION == 1
    serial = BACKEND.build_pipeline("/RVV/schedule.mlir")
    parallel = BACKEND.build_pipeline(
        "/RVV/schedule.mlir", par_sched_path="/RVV/parallel.mlir", perop_parallel=True
    )
    assert (
        hashlib.sha256(serial.encode()).hexdigest()
        == "7f7c9e0df67a937a73d357f9817975268a1255ae65c1b4000f6dd51f1c05aecb"
    )
    assert (
        hashlib.sha256(parallel.encode()).hexdigest()
        == "dcc6464b56db7ca402b0b736a74e3458d4cf37ed14a52a2e705cde73439b2875"
    )
    assert "transform-interpreter{entry-point=__transform_main}" in serial
    assert "convert-vector-to-llvm" in serial


def test_active_schedules_are_byte_identical_to_the_published_payloads():
    champion = ROOT / "schedules/impr_tuned_wholemodel_vf_int8/schedule.mlir"
    prepack = ROOT / "schedules/kodiak_prepack_w8a8_v1/schedule.mlir"
    assert hashlib.sha256(champion.read_bytes()).hexdigest() == (
        "928064a1264cf3dd4e70c1f6d162f989c3cc67160c36e2c19a94e731c10867b4"
    )
    assert prepack.read_bytes() == champion.read_bytes()
    assert BACKEND.schedule_path().name == BACKEND.DEFAULT_SCHEDULE


def test_multicore_schedule_variants_preserve_frozen_bytes():
    cases = (
        ({}, "43f9187c5d1d00ccdafc741116a266a3aad96cd61d247611697130f76a02420b"),
        (
            {"tile_aligned": True},
            "56897f03d76cbc3e4457698f0c55907a3c684d2d20de8ab8cfe3e70868c5197c",
        ),
        (
            {"chunks": []},
            "433d3ea18b98b3fedee2b2a0d1a49c51fc4df6cbb8a218b8b6112b149d980aa8",
        ),
    )
    for kwargs, expected in cases:
        actual = BACKEND.parallel_transform_schedule(2, **kwargs)
        assert hashlib.sha256(actual.encode()).hexdigest() == expected
    assert BACKEND.vec_pre_schedule(frozenset()) is None
