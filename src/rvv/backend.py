"""RVV provider binding, package resources and stable compatibility API.

The feature registry lives in features/registry.py; Transform entry placement
and ordered lowering passes live in pipelines.py. Native invocation and artifact
publication remain in the service adapter and Merlin's shared lowering transport.
Chip-bound registries and legacy whole-function hooks retain their public boundary.
"""

from __future__ import annotations

import hashlib
import sys
from importlib import import_module
from pathlib import Path

from merlin.compiler.passes import contract as schedule_contract
from merlin.compiler.passes import parallel as parallel_schedule

API_VERSION = 1

# Exact compatibility names; the implementation is shared below providers/drivers.
PARALLEL_ENTRY = parallel_schedule.PARALLEL_ENTRY
_PARALLEL_DIM_NUM_THREADS = parallel_schedule._PARALLEL_DIM_NUM_THREADS
_perop_parallel_schedule = parallel_schedule._perop_parallel_schedule
parallel_transform_schedule = parallel_schedule.parallel_transform_schedule
class_token = schedule_contract.class_token
par_tag_for = schedule_contract.par_tag_for
DEFAULT_SCHEDULE = "impr_tuned_wholemodel_vf_int8"
RVV_FLAGS = ("-march=rv64gcv", "-fno-vectorize", "-fno-slp-vectorize")
DEFAULT_CFLAGS = RVV_FLAGS
SPIKE_CFLAGS = (
    "-march=rv64gcv",
    "-mabi=lp64d",
    "-mcmodel=medany",
    "-O2",
    "-ffreestanding",
    "-fno-builtin",
)

# Deallocation and generic pass-list splicing remain shared Merlin infrastructure.
from merlin.compiler.selection.features import FeatureRegistry, ImprFeature

from . import pipelines

# Compatibility exports are exact objects from their canonical feature owner.
from .features import registry as feature_registry

register_accumulator_grids = feature_registry.register_accumulator_grids
_ensure_v3_kblocked_microkernel = feature_registry._ensure_v3_kblocked_microkernel
_ensure_v3_microkernel = feature_registry._ensure_v3_microkernel
_ensure_v3_perop_microkernel = feature_registry._ensure_v3_perop_microkernel
_ensure_v3_scalable_microkernel = feature_registry._ensure_v3_scalable_microkernel
_ensure_v3_unrolled_microkernel = feature_registry._ensure_v3_unrolled_microkernel
register_dynamic_accumulator_from_name = feature_registry.register_dynamic_accumulator_from_name
register_grid_features = feature_registry.register_grid_features
register_other_vector_features = feature_registry.register_other_vector_features
_ensure_perop_block = feature_registry._ensure_perop_block
_perop_mr_sentinel = feature_registry._perop_mr_sentinel
_perop_nr_sentinel = feature_registry._perop_nr_sentinel
register_perop_from_name = feature_registry.register_perop_from_name
register_perop_features = feature_registry.register_perop_features
ensure_vec_noncontraction = feature_registry.ensure_vec_noncontraction
vec_noncontraction_lanes = feature_registry.vec_noncontraction_lanes
register_vec_noncontraction = feature_registry.register_vec_noncontraction
register_vec_noncontraction_from_name = feature_registry.register_vec_noncontraction_from_name
_im2col_pack_feature = feature_registry._im2col_pack_feature
_weight_panel_feature = feature_registry._weight_panel_feature
WHOLEMODEL_VF_NAME = feature_registry.WHOLEMODEL_VF_NAME
fused_vfmacc_tiled_schedule = feature_registry.fused_vfmacc_tiled_schedule
wholemodel_pipeline = feature_registry.wholemodel_pipeline
wholemodel_schedule = feature_registry.wholemodel_schedule
wholemodel_vf_pipeline = feature_registry.wholemodel_vf_pipeline
wholemodel_vf_schedule = feature_registry.wholemodel_vf_schedule
register_conv_block_feature = feature_registry.register_conv_block_feature
register_requant_features = feature_registry.register_requant_features
_HISTORICAL_METADATA = feature_registry._HISTORICAL_METADATA
_RvvFeatureRegistry = feature_registry._RvvFeatureRegistry
_legacy_feature = feature_registry._legacy_feature
_FEATURES = feature_registry._FEATURES
normalize_features = feature_registry.normalize_features
vec_noncontraction_shape = feature_registry.vec_noncontraction_shape
ensure_parallel_grain = feature_registry.ensure_parallel_grain
ensure_parallel_team = feature_registry.ensure_parallel_team
parallel_grain_threshold = feature_registry.parallel_grain_threshold
parallel_team_work = feature_registry.parallel_team_work
perop_mr_sentinel = feature_registry.perop_mr_sentinel
perop_nr_sentinel = feature_registry.perop_nr_sentinel
parse_perop_mr_sentinel = feature_registry.parse_perop_mr_sentinel
parse_perop_nr_sentinel = feature_registry.parse_perop_nr_sentinel
ensure_perop_block = feature_registry.ensure_perop_block
ensure_v3_microkernel = feature_registry.ensure_v3_microkernel
ensure_v3_kblocked_microkernel = feature_registry.ensure_v3_kblocked_microkernel
ensure_v3_unrolled_microkernel = feature_registry.ensure_v3_unrolled_microkernel
ensure_v3_scalable_microkernel = feature_registry.ensure_v3_scalable_microkernel
ensure_v3_perop_microkernel = feature_registry.ensure_v3_perop_microkernel
unclaimed_op_classes = feature_registry.unclaimed_op_classes
accum_microkernel_marker = feature_registry.accum_microkernel_marker
accum_microkernel_rewrite_source = feature_registry.accum_microkernel_rewrite_source
accum_microkernel_runner_source = feature_registry.accum_microkernel_runner_source
apply_schedule = feature_registry.apply_schedule
apply_pipeline = feature_registry.apply_pipeline
apply_cflags = feature_registry.apply_cflags

_fuse_post = pipelines._fuse_post
_par_residue_parallel = pipelines._par_residue_parallel
VEC_PRE_ENTRY = pipelines.VEC_PRE_ENTRY
_VEC_PRE_SKELETON = pipelines._VEC_PRE_SKELETON
_vec_after_specialize = pipelines._vec_after_specialize
vec_pre_schedule = pipelines.vec_pre_schedule
build_pipeline = pipelines.build_pipeline
finalization_passes = pipelines.finalization_passes
deallocation_passes = pipelines.deallocation_passes
PipelineError = pipelines.PipelineError
splice_passes = pipelines.splice_passes


def prepare_build_bundle(
    src: Path | str, work: Path | str, features: frozenset[str] | None
) -> Path:
    """Apply package-owned weight layout preparation before RVV model lowering."""
    from .packing.prepack import prepare_build_bundle as prepare

    return prepare(src, work, normalize_features(features or frozenset()))


def vector_cflags(common: list[str] | tuple[str, ...]) -> list[str]:
    """Add the RVV ISA and controlled-vectorization policy to shared target flags."""
    return [*RVV_FLAGS, *common]


def spike_cflags() -> list[str]:
    """Frozen RVV bare-metal model flags for the Spike backend."""
    return list(SPIKE_CFLAGS)


def spike_harness_cflags(model_flags: list[str]) -> list[str]:
    """Use the selected ISA/ABI with GCC-compatible harness options."""
    flags = spike_cflags()
    for prefix in ("-march=", "-mabi=", "-mcmodel="):
        selected = [flag for flag in model_flags if flag.startswith(prefix)]
        if selected:
            flags = [flag for flag in flags if not flag.startswith(prefix)]
            flags.append(selected[-1])
    return flags


def schedule_path(name: str = DEFAULT_SCHEDULE) -> Path:
    """Resolve a schedule shipped by this exact provider revision."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"invalid RVV schedule name {name!r}")
    root = Path(__file__).resolve().parents[2] / "schedules" / name
    for file in ("manifest.yaml", "knobs.yaml", "schedule.mlir"):
        if not (root / file).is_file():
            raise FileNotFoundError(root / file)
    return root


def baseline_schedule_text() -> str:
    """Return the frozen, package-owned schedule used when a caller supplies none."""
    return (schedule_path() / "schedule.mlir").read_text(encoding="utf-8")


class BoundBackendV1:
    """One target's RVV backend with chip feature definitions isolated per build."""

    API_VERSION = 1

    def __init__(self, extension_features: tuple[ImprFeature, ...]):
        registry = FeatureRegistry()
        registry.entries.update(_FEATURES.entries)

        def register_base(name: str) -> bool:
            try:
                feature = _FEATURES.get(name)
            except KeyError:
                return False
            registry.register(feature)
            return True

        registry.lazy_register = register_base
        for feature in extension_features:
            if not isinstance(feature, ImprFeature):
                raise TypeError(
                    f"extension feature must be ImprFeature, got {type(feature).__name__}"
                )
            registry.register(feature)
        self.registry = registry

    def __getattr__(self, name: str):
        return getattr(sys.modules[__name__], name)

    def normalize_features(self, features) -> frozenset[str]:
        return self.registry.normalize(features)

    def feature_names(self) -> tuple[str, ...]:
        """Return registered feature names for a composed target."""
        return tuple(sorted(self.registry.entries))

    def apply_schedule(self, schedule_text: str, features: frozenset[str]) -> str:
        return self.registry.apply_schedule(schedule_text, features)

    def apply_pipeline(self, passes: list[str], features: frozenset[str]) -> list[str]:
        return self.registry.apply_pipeline(passes, features)

    def apply_cflags(self, cflags: list[str], features: frozenset[str]) -> list[str]:
        return self.registry.apply_cflags(cflags, features)

    def vec_pre_schedule(self, features: frozenset[str]) -> str | None:
        return vec_pre_schedule(features, _registry=self.registry)

    def build_pipeline(self, *args, **kwargs) -> str:
        return build_pipeline(*args, _registry=self.registry, **kwargs)

    def prepare_build_bundle(self, src: Path | str, work: Path | str, features) -> Path:
        from .packing.prepack import prepare_build_bundle as prepare

        return prepare(src, work, self.normalize_features(features or frozenset()))


def bind(extension_features: tuple[ImprFeature, ...] = ()) -> BoundBackendV1:
    """Compose chip-specific named features without mutating the shared RVV provider."""
    return BoundBackendV1(extension_features)


_LEGACY_MODULES = {
    "active_features": "transforms.active_features",
    "accumulator_grid": "features.accumulator_grid",
    "dynamic_accumulator": "features.dynamic_accumulator",
    "grid_features": "features.grid_features",
    "other_vector_features": "features.other_vector_features",
    "perop_features": "features.perop_features",
    "vector_features": "features.vector_features",
    "reduce_vec": "transforms.reduce_vec",
    "im2col_pack": "packing.im2col_pack",
    "weight_panel": "packing.weight_panel",
}


def legacy_module(name: str):
    """Resolve a previous flat module name for Merlin's compatibility facades."""
    try:
        path = _LEGACY_MODULES[name]
    except KeyError as exc:
        raise ValueError(f"unknown RVV compatibility module {name!r}") from exc
    return import_module(f".{path}", __package__)


def reduction_schedule_api():
    """Expose the RVV reduction Transform arms to Merlin's compatibility facade."""
    return legacy_module("other_vector_features")


def im2col_pack_api():
    """Return the RVV im2col panel packing implementation."""
    from .packing import im2col_pack

    return im2col_pack


def perop_blocks_api():
    """Return the RVV per-operation block and tagging implementation."""
    from .transforms import perop_blocks

    return perop_blocks


def perop_features_api():
    """Return RVV per-operation feature names and sentinel prefixes."""
    from .features import perop_features

    return perop_features


def microkernel_policy_api():
    """Return RVV shape-aware microkernel policy and schedule rendering."""
    from .policy import microkernel_policy

    return microkernel_policy


def blocking_cost_api():
    """Return RVV register-block legality and cost policy."""
    from .policy import blocking_cost

    return blocking_cost


def weight_prepack_api():
    """Return the RVV weight-layout prepack implementation."""
    from .packing import prepack

    return prepack


def requant_fuse_api():
    """Return the RVV contraction-requantization fusion implementation."""
    from .transforms import requant_fuse

    return requant_fuse


def reduce_vec_api():
    """Return the RVV reduction vectorizer implementation."""
    from .transforms import reduce_vec

    return reduce_vec


def default_perop_caps(vlen: int | None) -> tuple[int, int, int]:
    from .policy.perop_caps import default_perop_caps as caps

    return caps(vlen)


def frozen_blocks_api():
    """Return frozen RVV register-block descriptions."""
    from .policy import frozen_blocks

    return frozen_blocks


def shape_adapt_api():
    """Return RVV workload shape adaptation and pinned-block diagnostics."""
    from .policy import shape_adapt

    return shape_adapt


def shape_adapted_features(pkg, model_dir: Path | str, *, harts: int = 1) -> list[str]:
    from .policy.shape_adapt import shape_adapted_features as adapt

    return adapt(pkg, model_dir, harts=harts)


def blocking_risks(pkg, model_dir: Path | str) -> list[str]:
    from .policy.shape_adapt import blocking_risks as risks

    return risks(pkg, model_dir)


def lmul_group_api():
    """Return the RVV register-group derivation policy."""
    from .policy import lmul_group

    return lmul_group


LMUL_GROUP_SENTINEL = "lmul_register_group"


def lmul_group_feature(lmul: int) -> str:
    """Name the exact supported register-group width, refusing unsupported widths."""
    from .policy.lmul_group import LMUL_LADDER, LmulDerivationError

    if int(lmul) not in LMUL_LADDER:
        raise LmulDerivationError(f"LMUL={lmul!r} is not one of {LMUL_LADDER}")
    return f"lmul_group_m{int(lmul)}"


def default_vec_rank_shape() -> tuple[int, int]:
    """The published default lane count and maximum noncontraction rank."""
    from .features.vector_features import (
        VEC_NONCONTRACTION_LANES,
        VEC_NONCONTRACTION_MAX_RANK,
    )

    return VEC_NONCONTRACTION_LANES, VEC_NONCONTRACTION_MAX_RANK


def vec_rank_bounds() -> tuple[int, int]:
    """The admitted rank interval for bounded noncontraction vectorization."""
    from .features.vector_features import (
        VEC_NONCONTRACTION_MAX_RANK,
        VEC_NONCONTRACTION_MIN_RANK,
    )

    return VEC_NONCONTRACTION_MIN_RANK, VEC_NONCONTRACTION_MAX_RANK


def tag_vector_generics(module, *, lanes: int, max_rank: int) -> dict[str, int]:
    from .transforms.vector_tag import tag_vector_generics as tag

    return tag(module, lanes=lanes, max_rank=max_rank)


def vectorize_amax_reduction(module, *, lanes: int, max_rank: int) -> dict:
    from .features.vector_features import VEC_NONCONTRACTION_MIN_RANK
    from .transforms.reduce_vec import apply

    return apply(
        module, lanes=lanes, min_rank=VEC_NONCONTRACTION_MIN_RANK, max_rank=max_rank
    )


def weight_panel_api():
    """Return the RVV weight panel packing and ABI implementation."""
    from .packing import weight_panel

    return weight_panel


def render_schedule(knobs: dict) -> str:
    from .policy.microkernel_policy import render_schedule as render

    return render(knobs)


def _policy_source_paths(*, operation=None):
    """Actual provider and selected shared schedule bytes for source attribution.

    Includes the service/contract selection edges; this is not native toolchain
    identity or a complete transitive shared-library closure.
    """
    root = Path(__file__).resolve().parent
    package = root.parents[1]
    if operation == "blocking":
        owned = (root / "backend.py", root / "policy/blocking_cost.py")
    elif operation is None:
        owned = tuple(path for path in sorted(root.rglob("*"))
                      if path.is_file() and path.suffix in {".py", ".json"})
    else:
        raise ValueError("unknown vector policy source scope")
    edges = tuple(package / relative for relative in (
        "backend.py", "service.py", "provider.yaml", "contracts/target_contract.yaml",
    ) if (package / relative).is_file())
    shared = (() if operation == "blocking" else
              (Path(parallel_schedule.__file__), Path(schedule_contract.__file__)))
    return (*owned, *edges, *shared)


def policy_source_files(*, operation=None):
    """Refuse source drift since this provider's import before returning paths."""
    selected = _policy_source_paths(operation=operation)
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != _POLICY_IMPORT_SOURCES.get(str(path))
           for path in selected):
        raise ValueError("selected vector policy source changed after import; use a fresh provider process")
    return selected


def register_microkernel_resolver() -> None:
    from .policy.microkernel_policy import _register_rvv_microkernel_resolver

    _register_rvv_microkernel_resolver()


def microkernel_features(knobs: dict, shapes=()) -> list[str]:
    """Resolve with this provider's policy without changing ambient resolvers."""
    from merlin.kernels.microkernel import MicrokernelSpec

    from .policy.microkernel_policy import (
        _rvv_microkernel_resolver,
        _rvv_microkernel_shape_policy,
    )

    spec = MicrokernelSpec.from_knobs(knobs)
    return list(
        _rvv_microkernel_shape_policy(spec, tuple(shapes))
        if shapes
        else _rvv_microkernel_resolver(spec)
    )


# Snapshot provider-owned bytes at import. This detects later edits; it does not
# attest Python bytecode or a complete transitive/native dependency closure.
_POLICY_IMPORT_SOURCES = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in _policy_source_paths()}
