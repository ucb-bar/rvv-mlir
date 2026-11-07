"""RVV accumulator-resident v1/v2 register-tile feature grids."""

from __future__ import annotations

from merlin.llvmlower.feature_registry import FeatureRegistry, ImprFeature

from .active_features import (
    _accumulator_resident_pipeline,
    _accumulator_resident_pre_schedule,
    _accumulator_resident_v2_pipeline,
)

_GRID = ((4, 16, 16), (4, 16, 32), (8, 16, 16), (4, 32, 16))


def register_accumulator_grids(registry: FeatureRegistry) -> None:
    for mr, nr, kc in _GRID:
        name = (
            "accumulator_resident_microkernel"
            if (mr, nr, kc) == (4, 16, 16)
            else f"accum_resident_{mr}_{nr}_{kc}"
        )
        registry.register(
            ImprFeature(
                name=name,
                action_class="PASS",
                description=f"RVV post-bufferization accumulator hoist with tile {mr}x{nr}x{kc}.",
                edit_pipeline=_accumulator_resident_pipeline,
                edit_schedule=lambda _text, _mr=mr, _nr=nr, _kc=kc: (
                    _accumulator_resident_pre_schedule(_mr, _nr, _kc)
                ),
                schedule_replace=True,
            )
        )
    for name, kwargs in (
        ("accumulator_resident_ntail", {"NR_bmm": 8}),
        ("accumulator_resident_mtail", {"MR_mm": 1}),
    ):
        registry.register(
            ImprFeature(
                name=name,
                action_class="PASS",
                description=f"RVV accumulator-resident schedule with {name} tail clamp.",
                edit_pipeline=_accumulator_resident_pipeline,
                edit_schedule=lambda _text, _kw=kwargs: (
                    _accumulator_resident_pre_schedule(4, 16, 16, **_kw)
                ),
                schedule_replace=True,
            )
        )
    for mr, nr, kc in _GRID:
        name = (
            "accumulator_resident_v2"
            if (mr, nr, kc) == (4, 16, 16)
            else f"accum_resident_v2_{mr}_{nr}_{kc}"
        )
        registry.register(
            ImprFeature(
                name=name,
                action_class="PASS",
                description=f"RVV tensor-level accumulator hoist with tile {mr}x{nr}x{kc}.",
                edit_pipeline=_accumulator_resident_v2_pipeline,
                edit_schedule=lambda _text, _mr=mr, _nr=nr, _kc=kc: (
                    _accumulator_resident_pre_schedule(_mr, _nr, _kc)
                ),
                schedule_replace=True,
            )
        )
