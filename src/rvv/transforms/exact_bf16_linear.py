"""Explicit exact-order BF16 linear provider with portable region admission."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from rvv.policy.panel_geometry import PanelTarget
from rvv.transforms import exact_bf16_leaf as leaf
from rvv.transforms._compiler_flags import with_march

CFLAGS = (
    "-O2", "-march=rv64gcv", "-mabi=lp64d", "-ffp-contract=off",
    "-frounding-math", "-fno-fast-math",
)
INDEX_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class LinearSpec:
    """Static contiguous BF16 A[M,K], original W[N,K], and output[M,N]."""

    m: int
    k: int
    n: int
    symbol: str = "rvv_exact_bf16_linear"

    def __post_init__(self) -> None:
        for name in ("m", "k", "n"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= INDEX_MAX:
                raise ValueError(f"{name} must be a positive signed-RV64 integer")
        # Reserve a public namespace to avoid private identifier collisions.
        if not isinstance(self.symbol, str) or not re.fullmatch(
            r"rvv_exact_bf16_linear(?:_[A-Za-z0-9_]+)?", self.symbol
        ):
            raise ValueError("symbol must use the rvv_exact_bf16_linear namespace")
        if max(self.m * self.k * 2, self.n * self.k * 2, self.m * self.n * 4) > INDEX_MAX:
            raise ValueError("buffer byte span exceeds signed-RV64 indexing")


def _selected_leaf(spec: LinearSpec, enabled: bool, target: PanelTarget | None) -> leaf.LeafSpec | None:
    if type(enabled) is not bool:
        raise ValueError("enabled must be an explicit boolean")
    if not isinstance(spec, LinearSpec):
        raise TypeError("spec must be LinearSpec")
    if not enabled:
        return None
    return leaf.LeafSpec(spec.m, spec.k, spec.n, symbol=spec.symbol + "_leaf", target=target)


def emit_mlir(spec: LinearSpec, *, enabled: bool = False, target: PanelTarget | None = None) -> str | None:
    """Author a leaf with a sequential K loop and unflagged mulf/addf."""
    selected = _selected_leaf(spec, enabled, target)
    return leaf.emit_mlir(selected) if selected is not None else None


def emit_wrapper(spec: LinearSpec) -> str:
    """Region admission and transactional output around the ordinary MLIR ABI."""
    return f"""
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <limits.h>
#include <fenv.h>
#ifdef __FAST_MATH__
#error exact_bf16_linear_forbids_fast_math
#endif
_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 && FLT_EVAL_METHOD == 0, "IEEE binary32 required");
typedef struct {{ uint16_t *allocated, *aligned; int64_t offset, sizes[1], strides[1]; }} D;
extern int32_t _mlir_ciface_{spec.symbol}_leaf(D *, D *, D *);
static int finite_bf16(const uint16_t *p, size_t count) {{
  for (size_t i = 0; i < count; ++i)
    if ((p[i] & 0x7fffu) >= 0x7f80u) return 0;
  return 1;
}}
static int zero_seed(const unsigned char *p, size_t count) {{
  for (size_t i = 0; i < count; ++i)
    if (p[i]) return 0;
  return 1;
}}
int {spec.symbol}(void *context, const void *const *inputs, void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !outputs[0] || fegetround() != FE_TONEAREST) return -1;
  const size_t sizes[3] = {{{spec.m*spec.k*2}u, {spec.n*spec.k*2}u, {spec.m*spec.n*4}u}};
  const uintptr_t destination = (uintptr_t)outputs[0];
  if (destination > UINTPTR_MAX - {spec.m*spec.n*2}u) return -1;
  for (unsigned p = 0; p < 3; ++p) {{
    uintptr_t address = (uintptr_t)inputs[p];
    if (!address || address > UINTPTR_MAX - sizes[p] ||
        (address < destination + {spec.m*spec.n*2}u && destination < address + sizes[p])) return -1;
  }}
  if (!finite_bf16(inputs[0], {spec.m*spec.k}u) ||
      !finite_bf16(inputs[1], {spec.n*spec.k}u) || !zero_seed(inputs[2], {spec.m*spec.n*4}u)) return -2;
  uint16_t *temporary = malloc({spec.m*spec.n*2}u);
  if (!temporary) return -3;
  D a = {{(uint16_t *)inputs[0], (uint16_t *)inputs[0], 0, {{{spec.m*spec.k}}}, {{1}}}};
  D w = {{(uint16_t *)inputs[1], (uint16_t *)inputs[1], 0, {{{spec.n*spec.k}}}, {{1}}}};
  D o = {{temporary, temporary, 0, {{{spec.m*spec.n}}}, {{1}}}};
  int result = _mlir_ciface_{spec.symbol}_leaf(&a, &w, &o) ? -2 : 0;
  if (!result) memcpy(outputs[0], temporary, {spec.m*spec.n*2}u);
  free(temporary);
  return result;
}}
"""


def compile_kernel(
    spec: LinearSpec, *, work: str | Path, compiler: str | Path,
    mlir_opt: str | Path | None = None, mlir_translate: str | Path | None = None,
    clang: str | Path | None = None, llc: str | Path | None = None,
    enabled: bool = False,
    target: PanelTarget | None = None,
    timeout: int = 120, march: str = "rv64gcv",
) -> dict:
    """Compile an opt-in MLIR leaf and portable C wrapper with explicit tools."""
    selected = _selected_leaf(spec, enabled, target)
    flags = compiler_flags(march)
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    if selected is None:
        return {"enabled": False}
    if mlir_opt is None or mlir_translate is None or (clang is None) == (llc is None):
        raise ValueError("select mlir_opt, mlir_translate and exactly one of clang or llc explicitly")
    work = Path(work).absolute()
    work.mkdir(parents=True, exist_ok=False)
    codegen = ({"llc": llc} if llc is not None else {"clang": clang})
    lowered = leaf.compile_leaf(
        selected,
        work=work / "leaf", mlir_opt=mlir_opt, mlir_translate=mlir_translate,
        **codegen, timeout=timeout, march=march,
    )
    wrapper, wrapper_obj, obj = work / "wrapper.c", work / "wrapper.o", work / "kernel.o"
    wrapper.write_text(emit_wrapper(spec))
    commands = [
        [str(compiler), *flags, "-c", str(wrapper), "-o", str(wrapper_obj)],
        [str(compiler), f"-march={march}", "-mabi=lp64d", "-r",
         str(wrapper_obj), lowered["object"], "-o", str(obj)],
    ]
    for index, command in enumerate(commands):
        with (work / f"wrapper{index}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    report = dict(enabled=True, spec=asdict(spec), source=lowered["source"],
                  wrapper=str(wrapper), object=str(obj), leaf=lowered,
                  commands=lowered["commands"] + commands,
                  load_strategy="masked_strided_gather", lanes=lowered["lanes"], rows=lowered["rows"])
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compiler_flags(march: str = "rv64gcv") -> tuple[str, ...]:
    """Use shipped arithmetic flags with a caller-selected target ISA."""
    return with_march(CFLAGS, march)
