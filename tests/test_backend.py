"""Executable RVV provider: pass ordering and published schedules."""

import hashlib
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = types.ModuleType("rvv_provider")
PACKAGE.__path__ = [str(ROOT)]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location(
    "rvv_provider.backend", ROOT / "backend.py"
)
assert SPEC is not None and SPEC.loader is not None
BACKEND = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BACKEND
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
    assert BACKEND.DEFAULT_CFLAGS == (
        "-march=rv64gcv",
        "-fno-vectorize",
        "-fno-slp-vectorize",
    )
    assert BACKEND.vector_cflags(
        ["-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"]
    ) == [
        *BACKEND.DEFAULT_CFLAGS,
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
    ]
    assert BACKEND.spike_cflags() == [
        "-march=rv64gcv",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
    ]
    assert BACKEND.spike_harness_cflags(
        ["-march=rv64gcv_zvl256b", "-mabi=lp64d", "-mcmodel=medany"]
    ) == [
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-march=rv64gcv_zvl256b",
        "-mabi=lp64d",
        "-mcmodel=medany",
    ]
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
    assert BACKEND.baseline_schedule_text() == champion.read_text(encoding="utf-8")
    fp32 = (
        BACKEND.schedule_path("rvv_tuned_v1_d1_vfmacc_outerproduct") / "schedule.mlir"
    )
    assert hashlib.sha256(fp32.read_bytes()).hexdigest() == (
        "dec409ddd8a08eec90f4aefea8e4f73d5f02770b5d358bc6c340c73add9fc3c1"
    )
    assert BACKEND.schedule_path("hand_v0_int8").is_dir()


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


def test_published_wholemodel_feature_is_owned_here_and_matches_frozen_bytes():
    for names in (
        {"accumulator_resident_wholemodel_vf"},
        {"accumulator_resident_wholemodel_vf", "prepack_weight_layout"},
    ):
        features = BACKEND.normalize_features(names)
        schedule = BACKEND.apply_schedule(BACKEND.baseline_schedule_text(), features)
        pipeline = BACKEND.build_pipeline("/RVV/schedule.mlir", features=features)
        assert hashlib.sha256(schedule.encode()).hexdigest() == (
            "93f7ea831e998e27ef8c52c9f93d6c3ec5bc033db3fdf484e7cd14156a1fddac"
        )
        normalized = pipeline.replace(tempfile.gettempdir(), "/TMP")
        assert hashlib.sha256(normalized.encode()).hexdigest() == (
            "e76ed9039baa36dd06a551ebe5ea4ff6a5b359f0aebd5db2dff4c5955a4f1dcb"
        )


def test_prepack_off_does_not_read_or_rewrite_bundle(tmp_path):
    source = tmp_path / "bundle"
    source.mkdir()
    work = tmp_path / "work"
    assert BACKEND.prepare_build_bundle(source, work, frozenset()) == source.resolve()
    assert not work.exists()


def test_published_fused_vfmacc_schedule_is_provider_owned():
    features = BACKEND.normalize_features({"fused_vfmacc_tiled"})
    schedule = BACKEND.apply_schedule(BACKEND.baseline_schedule_text(), features)
    assert hashlib.sha256(schedule.encode()).hexdigest() == (
        "393312d4e20b63c4644d01bd1c5a2601b36ab9fd071766abbdf627761a6d9356"
    )


def test_published_accumulator_resident_wholemodel_is_provider_owned():
    features = BACKEND.normalize_features({"accumulator_resident_wholemodel"})
    schedule = BACKEND.apply_schedule(BACKEND.baseline_schedule_text(), features)
    assert hashlib.sha256(schedule.encode()).hexdigest() == (
        "ac776529966ab3ddae691639c9c228316630d2743b5fe721ae81e547ea8e2bbf"
    )
    passes = [
        "canonicalize",
        "one-shot-bufferize{bufferize-function-boundaries}",
        "convert-vector-to-scf",
    ]
    actual = BACKEND.apply_pipeline(passes, features)
    normalized = str(actual).replace(tempfile.gettempdir(), "/TMP")
    assert hashlib.sha256(normalized.encode()).hexdigest() == (
        "e1df4bc8342c9713f63aeef9b12a94955fe906c196c5547afdc8fa8ebc90633c"
    )
    assert BACKEND.normalize_features({"fuse_quantize_round_convert"}) == frozenset(
        {"fuse_quantize_round_convert"}
    )


def test_noncontraction_vector_arms_and_name_derived_points_keep_frozen_bytes():
    cases = (
        (
            "vectorize_non_contraction_generics",
            "ae2044e97143c4eccc8e6417d2666c8622dd3d6073fd7ad9ea6b1c3124fe972e",
        ),
        (
            "vectorize_non_contraction_generics_l16_r5",
            "2b9e77fd23a354fc55feac4be47c4bf8bd7c04a9faf9f0bca3094a571c5ad43f",
        ),
    )
    for name, expected in cases:
        features = BACKEND.normalize_features({name})
        assert "erase_self_copy" in features
        schedule = BACKEND.apply_schedule(BACKEND.baseline_schedule_text(), features)
        assert hashlib.sha256(schedule.encode()).hexdigest() == expected
