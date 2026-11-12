"""RVV feature identities, isolated base registry and dynamic registration.

Own named feature availability and edits; do not select a target or native tool.
Chip extensions bind a separate registry through the provider's BoundBackendV1.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from merlin.compiler.selection.features import FeatureRegistry, ImprFeature

from ..packing.im2col_pack import _feature as _im2col_pack_feature
from ..packing.weight_panel import _feature as _weight_panel_feature
from ..transforms.active_features import (
    WHOLEMODEL_VF_NAME,
    fused_vfmacc_tiled_schedule,
    wholemodel_pipeline,
    wholemodel_schedule,
    wholemodel_vf_pipeline,
    wholemodel_vf_schedule,
)
from ..transforms.perop_blocks import ensure_registered as register_conv_block_feature
from ..transforms.requant_fuse import ensure_registered as register_requant_features
from . import vector_features as vector_definitions
from .accumulator_grid import register_accumulator_grids
from .dynamic_accumulator import (
    ensure_v3_kblocked_microkernel as _ensure_v3_kblocked_microkernel,
)
from .dynamic_accumulator import (
    ensure_v3_microkernel as _ensure_v3_microkernel,
)
from .dynamic_accumulator import (
    ensure_v3_perop_microkernel as _ensure_v3_perop_microkernel,
)
from .dynamic_accumulator import (
    ensure_v3_scalable_microkernel as _ensure_v3_scalable_microkernel,
)
from .dynamic_accumulator import (
    ensure_v3_unrolled_microkernel as _ensure_v3_unrolled_microkernel,
)
from .dynamic_accumulator import (
    register_from_name as register_dynamic_accumulator_from_name,
)
from .grid_features import register_grid_features
from .other_vector_features import register_other_vector_features
from .perop_features import (
    ensure_perop_block as _ensure_perop_block,
)
from .perop_features import (
    perop_mr_sentinel as _perop_mr_sentinel,
)
from .perop_features import (
    perop_nr_sentinel as _perop_nr_sentinel,
)
from .perop_features import (
    register_from_name as register_perop_from_name,
)
from .perop_features import (
    register_perop_features,
)
from .vector_features import (
    register_base as register_vec_noncontraction,
)
from .vector_features import (
    register_from_name as register_vec_noncontraction_from_name,
)
from .vector_features import (
    vec_noncontraction_lanes,
)

ensure_vec_noncontraction = vector_definitions.ensure_vec_noncontraction

_HISTORICAL_METADATA = json.loads(
    (Path(__file__).parent / "feature_metadata_v1.json").read_text(
        encoding="utf-8"
    )
)


class _RvvFeatureRegistry(FeatureRegistry):
    """Keep historical feature descriptions while implementations move to this provider."""

    def register(self, feature: ImprFeature) -> ImprFeature:
        if feature.name in _HISTORICAL_METADATA:
            feature = replace(
                feature,
                description=_HISTORICAL_METADATA[feature.name]["description"],
            )
        return super().register(feature)


def _legacy_feature(name: str) -> bool:
    """Migration bridge for experimental features not yet extracted to this provider."""
    if register_vec_noncontraction_from_name(_FEATURES, name):
        return True
    if register_perop_from_name(_FEATURES, name):
        return True
    if register_dynamic_accumulator_from_name(_FEATURES, name):
        return True
    from merlin.compiler.selection.shared import get_shared_feature

    try:
        shared = get_shared_feature(name)
    except KeyError:
        pass
    else:
        _FEATURES.register(shared)
        return True
    return False


_FEATURES = _RvvFeatureRegistry(lazy_register=_legacy_feature)
register_vec_noncontraction(_FEATURES)
register_grid_features(_FEATURES)
register_accumulator_grids(_FEATURES)
register_other_vector_features(_FEATURES)
register_perop_features(_FEATURES)
register_conv_block_feature(_FEATURES)
register_requant_features(_FEATURES)
_FEATURES.register(_im2col_pack_feature())
_FEATURES.register(_weight_panel_feature())
_FEATURES.register(
    ImprFeature(
        name="approximate_transcendental_activation",
        action_class="PASS",
        description="Target-neutral activation polynomial rewrite before target lowering.",
    )
)
_FEATURES.register(
    ImprFeature(
        name="erase_self_copy",
        action_class="PASS",
        description="Target-neutral self-copy erase in the shared post-bufferization runner.",
    )
)
_FEATURES.register(
    ImprFeature(
        name="accumulator_resident_wholemodel",
        action_class="PASS",
        description="Whole-model accumulator-resident RVV vector schedule and lowering.",
        edit_pipeline=wholemodel_pipeline,
        edit_schedule=lambda _text: wholemodel_schedule(),
        schedule_replace=True,
    )
)
_FEATURES.register(
    ImprFeature(
        name=WHOLEMODEL_VF_NAME,
        action_class="PASS",
        description="Published whole-model-safe vfmacc.vf accumulator-resident schedule and lowering.",
        edit_pipeline=wholemodel_vf_pipeline,
        edit_schedule=lambda _text: wholemodel_vf_schedule(),
        schedule_replace=True,
    )
)
_FEATURES.register(
    ImprFeature(
        name="prepack_weight_layout",
        action_class="PASS",
        description="Weight-layout preparation selected before RVV pipeline construction.",
    )
)
_FEATURES.register(
    ImprFeature(
        name="fuse_quantize_round_convert",
        action_class="PASS",
        description="Target-neutral exact quantization rewrite performed before target lowering.",
    )
)
_FEATURES.register(
    ImprFeature(
        name="fuse_quantize_round_convert_vec",
        action_class="PASS",
        description="Compose exact quantize rewrite with bounded RVV non-contraction vectorization.",
        implies=frozenset(
            {"fuse_quantize_round_convert", "vectorize_non_contraction_generics"}
        ),
    )
)
_FEATURES.register(
    ImprFeature(
        name="fused_vfmacc_tiled",
        action_class="PASS",
        description="Bounded tiled RVV fused-vfmacc Transform schedule.",
        edit_schedule=lambda _text: fused_vfmacc_tiled_schedule(),
        schedule_replace=True,
    )
)
_FEATURES.register(
    ImprFeature(
        name="intrinsic_microkernel",
        action_class="CODEGEN",
        description="CEILING REFERENCE (hand-written RVV intrinsic microkernel, not compiler-emitted); no codegen edit.",
    )
)


def normalize_features(features) -> frozenset[str]:
    return _FEATURES.normalize(features)


def vec_noncontraction_shape(features) -> tuple[int | None, int | None]:
    from .vector_features import vec_noncontraction_max_rank

    return vec_noncontraction_lanes(features), vec_noncontraction_max_rank(features)


def ensure_parallel_grain(threshold: int) -> str:
    from merlin.compiler.lower.model.parallel_grain import ensure_registered

    name = ensure_registered(threshold)
    _FEATURES.get(name)
    return name


def ensure_parallel_team(work_per_thread: int) -> str:
    from merlin.compiler.lower.model.parallel_team import ensure_registered

    name = ensure_registered(work_per_thread)
    _FEATURES.get(name)
    return name


def parallel_grain_threshold(features) -> int | None:
    from merlin.compiler.lower.model.parallel_grain import threshold_of

    return threshold_of(features)


def parallel_team_work(features) -> int | None:
    from merlin.compiler.lower.model.parallel_team import work_of

    return work_of(features)


def perop_mr_sentinel(mr_cap: int) -> str:
    return _perop_mr_sentinel(_FEATURES, mr_cap)


def perop_nr_sentinel(nr_cap: int) -> str:
    return _perop_nr_sentinel(_FEATURES, nr_cap)


def parse_perop_mr_sentinel(name: str) -> int | None:
    from .perop_features import parse_perop_mr_sentinel as parse

    return parse(name)


def parse_perop_nr_sentinel(name: str) -> int | None:
    from .perop_features import parse_perop_nr_sentinel as parse

    return parse(name)


def ensure_perop_block(table, kc: int, pairs=(), vec_epilogue: bool = False) -> str:
    return _ensure_perop_block(_FEATURES, table, kc, pairs, vec_epilogue)


def ensure_v3_microkernel(mr: int, nr: int, kc: int) -> str:
    return _ensure_v3_microkernel(_FEATURES, mr, nr, kc)


def ensure_v3_kblocked_microkernel(mr: int, nr: int, kc: int) -> str:
    return _ensure_v3_kblocked_microkernel(_FEATURES, mr, nr, kc)


def ensure_v3_unrolled_microkernel(mr: int, nr: int, kc: int) -> str:
    return _ensure_v3_unrolled_microkernel(_FEATURES, mr, nr, kc)


def ensure_v3_scalable_microkernel(mr: int, nr: int, kc: int) -> str:
    return _ensure_v3_scalable_microkernel(_FEATURES, mr, nr, kc)


def ensure_v3_perop_microkernel(
    mr_mm: int | None,
    nr_mm: int | None,
    mr_bmm: int | None,
    nr_bmm: int | None,
    kc: int,
) -> str:
    return _ensure_v3_perop_microkernel(_FEATURES, mr_mm, nr_mm, mr_bmm, nr_bmm, kc)


def unclaimed_op_classes(feature: str) -> list[str]:
    from .dynamic_accumulator import unclaimed_op_classes as unclaimed

    return unclaimed(feature)


def accum_microkernel_marker() -> str:
    from ..transforms.accum_microkernel import SCALARIZE_MARKER

    return SCALARIZE_MARKER


def accum_microkernel_rewrite_source() -> str:
    from ..transforms.accum_microkernel import rewrite_source

    return rewrite_source()


def accum_microkernel_runner_source(*, tag_bmm_tails: bool = False) -> str:
    from ..transforms.accum_microkernel import run_source

    return run_source(tag_bmm_tails=tag_bmm_tails)


def feature_definitions() -> tuple[ImprFeature, ...]:
    """Features a Merlin build resolves from this provider by name.

    The register-group widths, the direct-conv block request, the fused requantization
    epilogue and the panel-packed weight layout are implemented here; Merlin's shared
    optimization registry declares none of them and resolves each from this list. Each entry is
    its implementing module's own declaration, independent of this registry's historical copy.
    """
    from ..packing.weight_panel import _feature as weight_panel_feature
    from ..transforms.perop_blocks import _feature as conv_arm_feature
    from ..transforms.requant_fuse import FEATURE as REQUANT_FEATURE
    from ..transforms.requant_fuse import VEC_FEATURE as REQUANT_VEC_FEATURE
    from ..transforms.requant_fuse import _feature as requant_feature
    from .grid_features import lmul_group_features

    return (
        *lmul_group_features(),
        conv_arm_feature(),
        requant_feature(REQUANT_FEATURE),
        requant_feature(REQUANT_VEC_FEATURE),
        weight_panel_feature(),
    )


def apply_schedule(schedule_text: str, features: frozenset[str]) -> str:
    return _FEATURES.apply_schedule(schedule_text, features)


def apply_pipeline(passes: list[str], features: frozenset[str]) -> list[str]:
    return _FEATURES.apply_pipeline(passes, features)


def apply_cflags(cflags: list[str], features: frozenset[str]) -> list[str]:
    return _FEATURES.apply_cflags(cflags, features)


