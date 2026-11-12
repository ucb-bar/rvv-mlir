"""Ordinary MLIR leaf for borrowed BF16 buffers and ordered F32 arithmetic.

The region wrapper screens inputs and RNE mode and gives this leaf private
output storage. Callers supply valid buffer capacities. A nonzero return
indicates a nonfinite F32 accumulator or BF16 result; the wrapper must discard
that private output in that case.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from merlin.compiler.lower.llvm.passes import finalization_passes
from rvv.policy.panel_geometry import PanelTarget, derive
from rvv.transforms._compiler_flags import with_march

# No contraction schedule, reduction vectorization, BF16 extension or fast math.
PIPELINE = ",".join((
    "canonicalize", "cse", "convert-vector-to-scf", "convert-scf-to-cf",
    "expand-strided-metadata", "lower-affine",
    *finalization_passes(vector=True),
))
CODEGEN_FLAGS = ("--target=riscv64-unknown-elf", "-march=rv64gcv", "-mabi=lp64d",
                 "-O2", "-ffp-contract=off",
                 "-fno-fast-math", "-fno-vectorize",
                 "-fno-slp-vectorize")


@dataclass(frozen=True)
class LeafSpec:
    m: int
    k: int
    n: int
    symbol: str = "rvv_exact_bf16_mlir_leaf"
    target: PanelTarget | None = None
    weight_layout: str = "NK"

    @property
    def geometry(self):
        target = self.target or PanelTarget(128, 64, 16, 32, 1, 4*self.k*self.n)
        if (target.operand_bits, target.accumulator_bits) != (16, 32):
            raise ValueError("BF16 leaf requires 16-bit operands and 32-bit accumulators")
        return derive(target, k=self.k, output_width=self.n)

    @property
    def lanes(self):
        return self.geometry.nr

    def __post_init__(self):
        if self.weight_layout not in ("NK", "KN"):
            raise ValueError("weight_layout must be NK or KN")
        for name in ("m", "k", "n"):
            v = getattr(self, name)
            if type(v) is not int or not 0 < v <= (1 << 63) - 1:
                raise ValueError(f"{name} must be a positive signed-RV64 integer")
        self.geometry
        if max(self.m*self.k*2, self.n*self.k*2, self.m*self.n*2) > (1 << 63)-1:
            raise ValueError("buffer span exceeds signed-RV64 indexing")
        if not isinstance(self.symbol, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.symbol):
            raise ValueError("symbol must be an ASCII identifier")


def emit_mlir(spec: LeafSpec) -> str:
    """Return a destination-passing MLIR module, with i32 numeric-failure status.

    C interface: int32_t _mlir_ciface_SYMBOL(D1_i16*, D1_i16*, D1_i16*).
    D1_i16 is {uint16_t *allocated, *aligned; int64_t offset, sizes[1], strides[1];}.
    Shapes/strides are static contiguous; offsets are dynamic. Inputs may alias;
    the private destination must be disjoint. K executes in ascending order.
    """
    if not isinstance(spec, LeafSpec):
        raise TypeError("LeafSpec is required")
    m,k,n,l = spec.m,spec.k,spec.n,spec.lanes
    a,w,o = (f"memref<{s}xi16, strided<[1], offset: ?>>" for s in (m*k,n*k,m*n))
    vi,v16,v32,vf,vb = (f"vector<{l}x{t}>" for t in ("index","i16","i32","f32","i1"))
    lanes = ", ".join(str(x) for x in range(l))
    head = f'''module {{
  func.func @{spec.symbol}(%a: {a}, %w: {w}, %out: {o}) -> i32 attributes {{llvm.emit_c_interface}} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cm = arith.constant {m} : index
    %ck = arith.constant {k} : index
    %cn = arith.constant {n} : index
    %cl = arith.constant {l} : index
    %cr = arith.constant {spec.geometry.mr} : index
    %cfull = arith.constant {m // spec.geometry.mr * spec.geometry.mr} : index
    %false = arith.constant false
    %lane = arith.constant dense<[{lanes}]> : {vi}
    %vk = vector.splat %ck : {vi}
    %vn = vector.splat %cn : {vi}
    %zero16 = arith.constant dense<0> : {v16}
    %zerof = arith.constant dense<0.0> : {vf}
    %shift = arith.constant 16 : i32
    %vshift = arith.constant dense<16> : {v32}
    %vone = arith.constant dense<1> : {v32}
    %bias = arith.constant dense<32767> : {v32}
    %expmask = arith.constant dense<2139095040> : {v32}
    %bfmask = arith.constant dense<32767> : {v16}
    %bfinf = arith.constant dense<32640> : {v16}
'''
    blocks = []
    for tag, rows, lo, hi, step, initial in (
        ('full', spec.geometry.mr, '%c0', '%cfull', '%cr', '%false'),
        ('tail', 1, '%cfull', '%cm', '%c1', '%fullbad'),
    ):
        blocks.append(f'''    %{tag}bad = scf.for %mi = {lo} to {hi} step {step} iter_args(%mbad = {initial}) -> (i1) {{
''')
        for r in range(rows):
            blocks.append(f'''      %rowoff{r} = arith.constant {r} : index
      %row{r} = arith.addi %mi, %rowoff{r} : index
      %abase{r} = arith.muli %row{r}, %ck : index
      %obase{r} = arith.muli %row{r}, %cn : index
''')
        blocks.append(f'''      %nbad = scf.for %ni = %c0 to %cn step %cl iter_args(%bad = %mbad) -> (i1) {{
        %nv = vector.splat %ni : {vi}
        %columns = arith.addi %nv, %lane : {vi}
        %mask = arith.cmpi ult, %columns, %vn : {vi}
''')
        if spec.weight_layout == "NK":
            blocks.append(f'        %wbase = arith.muli %columns, %vk : {vi}\n')
        accs = ', '.join(f'%acc{r}' for r in range(rows))
        args = ', '.join(f'%sum{r} = %zerof' for r in range(rows))
        types = ', '.join([vf]*rows)
        blocks.append(f'        {accs} = scf.for %ki = %c0 to %ck step %c1 iter_args({args}) -> ({types}) {{\n')
        blocks.append(f'          %kv = vector.splat %ki : {vi}\n')
        if spec.weight_layout == "NK":
            blocks.append(f'          %wi = arith.addi %wbase, %kv : {vi}\n')
        else:
            blocks.append(f'''          %kn = arith.muli %kv, %vn : {vi}
          %wi = arith.addi %columns, %kn : {vi}
''')
        blocks.append(f'''          %wx = vector.gather %w[%c0][%wi], %mask, %zero16 : {w}, {vi}, {vb}, {v16} into {v16}
          %wb = arith.extui %wx : {v16} to {v32}
          %ws = arith.shli %wb, %vshift : {v32}
          %wf = arith.bitcast %ws : {v32} to {vf}
''')
        for r in range(rows):
            blocks.append(f'''          %ai{r} = arith.addi %abase{r}, %ki : index
          %ax{r} = memref.load %a[%ai{r}] : {a}
          %abits{r} = arith.extui %ax{r} : i16 to i32
          %ashift{r} = arith.shli %abits{r}, %shift : i32
          %af{r} = arith.bitcast %ashift{r} : i32 to f32
          %av{r} = vector.splat %af{r} : {vf}
          %product{r} = arith.mulf %av{r}, %wf : {vf}
          %next{r} = arith.addf %sum{r}, %product{r} : {vf}
''')
        values = ', '.join(f'%next{r}' for r in range(rows))
        blocks.append(f'          scf.yield {values} : {types}\n        }}\n')
        for r in range(rows):
            # Each row has a separate accumulator and one final BF16 rounding.
            previous_bad = "%bad" if r == 0 else f"%newbad{r-1}"
            blocks.append(f'''        %bits{r} = arith.bitcast %acc{r} : {vf} to {v32}
        %exp{r} = arith.andi %bits{r}, %expmask : {v32}
        %nonfinite{r} = arith.cmpi eq, %exp{r}, %expmask : {v32}
        %upper{r} = arith.shrui %bits{r}, %vshift : {v32}
        %lsb{r} = arith.andi %upper{r}, %vone : {v32}
        %biased{r} = arith.addi %bits{r}, %bias : {v32}
        %rounded{r} = arith.addi %biased{r}, %lsb{r} : {v32}
        %hi{r} = arith.shrui %rounded{r}, %vshift : {v32}
        %bf{r} = arith.trunci %hi{r} : {v32} to {v16}
        %magnitude{r} = arith.andi %bf{r}, %bfmask : {v16}
        %overflow{r} = arith.cmpi uge, %magnitude{r}, %bfinf : {v16}
        %failed{r} = arith.ori %nonfinite{r}, %overflow{r} : {vb}
        %activefailed{r} = arith.andi %failed{r}, %mask : {vb}
        %anyfailed{r} = vector.reduction <or>, %activefailed{r} : {vb} into i1
        %newbad{r} = arith.ori {previous_bad}, %anyfailed{r} : i1
        %oi{r} = arith.addi %obase{r}, %ni : index
        vector.maskedstore %out[%oi{r}], %mask, %bf{r} : {o}, {vb}, {v16}
''')
        blocks.append(f'''        scf.yield %newbad{rows-1} : i1
      }}
      scf.yield %nbad : i1
    }}
''')
    return head + ''.join(blocks) + '''    %status = arith.extui %tailbad : i1 to i32
    return %status : i32
  }
}
'''


def compile_leaf(spec: LeafSpec, *, work: str | Path, mlir_opt: str | Path,
                 mlir_translate: str | Path, clang: str | Path | None = None,
                 llc: str | Path | None = None, timeout: int = 120, march: str = "rv64gcv") -> dict:
    """Compile ordinary MLIR using explicit tools; create a fresh work directory."""
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be positive")
    flags = with_march(CODEGEN_FLAGS, march)
    if sum(x is not None for x in (clang, llc)) != 1:
        raise ValueError("select exactly one of clang or llc")
    source = emit_mlir(spec)
    minimums = [int(bits) for bits in re.findall(r"(?:^|_)zvl([0-9]+)b(?=_|$)", march)]
    if len(minimums) > 1 or any(bits < 128 or bits > 65536 or bits & (bits-1)
                                for bits in minimums):
        raise ValueError("march requires one valid minimum VLEN extension")
    target_vlen = spec.target.vlen_bits if spec.target is not None else 128
    minimum_vlen = minimums[0] if minimums else 128
    if minimum_vlen < target_vlen:
        raise ValueError(f"march minimum VLEN {minimum_vlen} is below target VLEN {target_vlen}")
    if llc is not None and not re.fullmatch(r"rv64gcv(?:_zvl[0-9]+b)?", march):
        raise ValueError("llc supports rv64gcv with an optional minimum VLEN; select clang for another ISA")
    work = Path(work).absolute()
    work.mkdir(parents=True, exist_ok=False)
    src, lowered, ll, obj = (work / s for s in ("leaf.mlir", "leaf.llvm.mlir", "leaf.ll", "leaf.o"))
    src.write_text(source)
    codegen = ([str(clang), *flags, "-c", str(ll), "-o", str(obj)]
               if clang is not None else
               [str(llc), "-mtriple=riscv64-unknown-elf", "-mattr=+m,+a,+f,+d,+c,+v" + (f",+zvl{minimum_vlen}b" if minimums else ""),
                "-target-abi=lp64d", "-filetype=obj", "-O2", "-fp-contract=off", str(ll), "-o", str(obj)])
    commands = [
        [str(mlir_opt), str(src), "--pass-pipeline=builtin.module("+PIPELINE+")", "-o", str(lowered)],
        [str(mlir_translate), "--mlir-to-llvmir", str(lowered), "-o", str(ll)],
        codegen,
    ]
    for index, command in enumerate(commands):
        if index == 2:
            llvm = ll.read_text()
            if re.search(r"\b(?:fmul|fadd)\s+(?:fast|nnan|ninf|nsz|arcp|contract|afn|reassoc)\b|llvm\.(?:fma|fmuladd)", llvm):
                raise RuntimeError("exact BF16 leaf acquired contraction/reassociation")
        with (work / f"stage{index}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    return {"source": str(src), "llvm_dialect": str(lowered), "llvm_ir": str(ll),
            "object": str(obj), "symbol": spec.symbol,
            "c_interface_symbol": "_mlir_ciface_"+spec.symbol,
            "pipeline": PIPELINE, "commands": commands, "lanes": spec.lanes,
            "rows": spec.geometry.mr,
            "reduction_order": "ascending_k", "weight_layout": spec.weight_layout,
            "contraction": False, "rounding": "integer_bf16_rne"}
