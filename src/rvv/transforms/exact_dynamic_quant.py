"""Opt-in integer-only MLIR lowering of generated BF16 dynamic quantization.

Preserves finite absolute maximum, group exponent, and FP8 RNE bit encoding.
No target routing, weight transformation, or floating-point arithmetic is added.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rvv.transforms._compiler_flags import with_march
from rvv.transforms.exact_bf16_leaf import CODEGEN_FLAGS, PIPELINE


@dataclass(frozen=True)
class QuantSpec:
    """Static contiguous groups of BF16 elements with one I32 exponent per group.

    The default eight I32 lanes fill one VLEN 256 register at LMUL 1, the
    measured target, so compile_kernel defaults to rv64gcv_zvl256b.
    """

    groups: int
    group_elements: int
    lanes: int = 8
    symbol: str = "rvv_exact_dynamic_quant"

    def __post_init__(self):
        for name in ("groups", "group_elements", "lanes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if self.lanes > 32 or self.lanes & (self.lanes - 1):
            raise ValueError("lanes must be a power of two up to 32")
        if max(self.groups * self.group_elements * 2, self.groups * 4) > (1 << 63) - 1:
            raise ValueError("buffer span exceeds signed RV64 indexing")
        if not isinstance(self.symbol, str) or not re.fullmatch(r"rvv_exact_dynamic_quant(?:_[A-Za-z0-9_]+)?", self.symbol):
            raise ValueError("symbol must use the rvv_exact_dynamic_quant namespace")


def _admit(spec, enabled):
    if enabled is not True:
        raise ValueError("exact dynamic quantization lowering requires enabled=True")
    if not isinstance(spec, QuantSpec):
        raise TypeError("QuantSpec required")


def _name(value):
    return str(value).replace("-", "neg")


def emit_mlir(s: QuantSpec, *, enabled: bool = False) -> str:
    """Emit destination-passing MLIR with masked tails and unaligned accesses.

    C interface is void _mlir_ciface_SYMBOL_leaf(D1_i8*, D1_i8*, D1_i8*).
    Source, codes, and exponents have distinct valid storage, as in emit_wrapper.
    """
    _admit(s, enabled)
    n, l = s.groups * s.group_elements, s.lanes
    x = f"memref<{2*n}xi8, strided<[1], offset: ?>>"
    y = f"memref<{n}xi8, strided<[1], offset: ?>>"
    e = f"memref<{4*s.groups}xi8, strided<[1], offset: ?>>"
    vi, vb, v16, v32, v8 = (f"vector<{l}x{t}>" for t in ("index", "i1", "i16", "i32", "i8"))
    vbytes, vmaskbytes = f"vector<{2*l}xi8>", f"vector<{2*l}xi1>"
    lines = [f'''module {{
 func.func @{s.symbol}_leaf(%x: {x}, %y: {y}, %eout: {e}) attributes {{llvm.emit_c_interface}} {{
  %z = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %four = arith.constant 4 : index
  %groups = arith.constant {s.groups} : index
  %count = arith.constant {s.group_elements} : index
  %lanes = arith.constant {l} : index
  %zerobytes = arith.constant dense<0> : {vbytes}
''']
    for k in (0, 1, 3, 4, 7, 8, 15, 16, 126, 127, 128, 254, 255, 32640, 32767, -6):
        lines.append(f'  %v{_name(k)} = arith.constant dense<{k}> : {v32}\n')
    for k in (0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 127, 135, 141, -127):
        lines.append(f'  %c{_name(k)} = arith.constant {k} : i32\n')
    lines.append(f'''  scf.for %g = %z to %groups step %one {{
   %base = arith.muli %g, %count : index
   %maxv = scf.for %i = %z to %count step %lanes iter_args(%acc = %v0) -> ({v32}) {{
    %left = arith.subi %count, %i : index
    %mask = vector.create_mask %left : {vb}
    %ix = arith.addi %base, %i : index
    %leftbytes = arith.muli %left, %two : index
    %maskbytes = vector.create_mask %leftbytes : {vmaskbytes}
    %ibytes = arith.muli %ix, %two : index
    %rawbytes = vector.maskedload %x[%ibytes], %maskbytes, %zerobytes : {x}, {vmaskbytes}, {vbytes} into {vbytes}
    %raw = vector.bitcast %rawbytes : {vbytes} to {v16}
    %bits = arith.extui %raw : {v16} to {v32}
    %mag = arith.andi %bits, %v32767 : {v32}
    %ex = arith.andi %mag, %v32640 : {v32}
    %finite = arith.cmpi ne, %ex, %v32640 : {v32}
    %value = arith.select %finite, %mag, %v0 : {vb}, {v32}
    %next = arith.maxui %acc, %value : {v32}
    scf.yield %next : {v32}
   }}
   %maximum = vector.reduction <maxui>, %maxv : {v32} into i32
   %high = arith.shrui %maximum, %c7 : i32
   %normal = arith.subi %high, %c135 : i32
''')
    prev = '%c0'
    for k in range(1, 7):
        lines.append(f'   %bit{k} = arith.cmpi uge, %maximum, %c{1 << k} : i32\n')
        lines.append(f'   %top{k} = arith.select %bit{k}, %c{k}, {prev} : i32\n')
        prev = f'%top{k}'
    lines.append(f'''   %subexp = arith.subi {prev}, %c141 : i32
   %nonzero = arith.cmpi ne, %maximum, %c0 : i32
   %sub = arith.select %nonzero, %subexp, %c0 : i32
   %isnormal = arith.cmpi ne, %high, %c0 : i32
   %exponent = arith.select %isnormal, %normal, %sub : i32
   %eindex = arith.muli %g, %four : index
   %evector = vector.broadcast %exponent : i32 to vector<1xi32>
   %ebytes = vector.bitcast %evector : vector<1xi32> to vector<4xi8>
   vector.store %ebytes, %eout[%eindex] {{alignment = 1 : i64}} : {e}, vector<4xi8>

   %lo = arith.maxsi %exponent, %cneg127 : i32
   %shift = arith.minsi %lo, %c127 : i32
   %vs = vector.splat %shift : {v32}
   scf.for %j = %z to %count step %lanes {{
    %remaining = arith.subi %count, %j : index
    %m = vector.create_mask %remaining : {vb}
    %jx = arith.addi %base, %j : index
    %remainingbytes = arith.muli %remaining, %two : index
    %mbytes = vector.create_mask %remainingbytes : {vmaskbytes}
    %jbytes = arith.muli %jx, %two : index
    %rbytes = vector.maskedload %x[%jbytes], %mbytes, %zerobytes : {x}, {vmaskbytes}, {vbytes} into {vbytes}
    %r = vector.bitcast %rbytes : {vbytes} to {v16}
    %b = arith.extui %r : {v16} to {v32}
    %sign = arith.shrui %b, %v15 : {v32}
    %eh = arith.shrui %b, %v7 : {v32}
    %exponentbits = arith.andi %eh, %v255 : {v32}
    %mant = arith.andi %b, %v127 : {v32}
    %negative = arith.cmpi ne, %sign, %v0 : {v32}
    %saturation = arith.select %negative, %v254, %v126 : {vb}, {v32}
    %sig = arith.ori %v128, %mant : {v32}
    %trunc = arith.shrui %sig, %v4 : {v32}
    %tail = arith.andi %sig, %v15 : {v32}
    %above = arith.cmpi ugt, %tail, %v8 : {v32}
    %tie = arith.cmpi eq, %tail, %v8 : {v32}
    %lsb = arith.andi %trunc, %v1 : {v32}
    %odd = arith.cmpi ne, %lsb, %v0 : {v32}
    %upontie = arith.andi %tie, %odd : {vb}
    %up = arith.ori %above, %upontie : {vb}
    %inc = arith.extui %up : {vb} to {v32}
    %rounded = arith.addi %trunc, %inc : {v32}
    %carry = arith.cmpi eq, %rounded, %v16 : {v32}
    %carry32 = arith.extui %carry : {vb} to {v32}
    %unbias = arith.subi %exponentbits, %v127 : {v32}
    %scaled = arith.subi %unbias, %vs : {v32}
    %final = arith.addi %scaled, %carry32 : {v32}
    %rebias = arith.addi %final, %v7 : {v32}
    %expnibble = arith.andi %rebias, %v15 : {v32}
    %expcode = arith.shli %expnibble, %v3 : {v32}
    %frac = arith.subi %rounded, %v8 : {v32}
    %fracmask = arith.andi %frac, %v7 : {v32}
    %frcode = arith.select %carry, %v0, %fracmask : {vb}, {v32}
    %sgcode = arith.shli %sign, %v7 : {v32}
    %body = arith.ori %expcode, %frcode : {v32}
    %code = arith.ori %sgcode, %body : {v32}
    %magnitude = arith.andi %code, %v127 : {v32}
    %reserved = arith.cmpi eq, %magnitude, %v127 : {v32}
    %reservedfix = arith.select %reserved, %saturation, %code : {vb}, {v32}
    %overflow = arith.cmpi sgt, %final, %v8 : {v32}
    %underflow = arith.cmpi slt, %final, %vneg6 : {v32}
    %ofix = arith.select %overflow, %saturation, %reservedfix : {vb}, {v32}
    %ufix = arith.select %underflow, %v0, %ofix : {vb}, {v32}
    %allones = arith.cmpi eq, %exponentbits, %v255 : {v32}
    %mantzero = arith.cmpi eq, %mant, %v0 : {v32}
    %inf = arith.andi %allones, %mantzero : {vb}
    %inffix = arith.select %inf, %saturation, %ufix : {vb}, {v32}
    %expzero = arith.cmpi eq, %exponentbits, %v0 : {v32}
    %mantnonzero = arith.cmpi ne, %mant, %v0 : {v32}
    %nan = arith.andi %allones, %mantnonzero : {vb}
    %clear = arith.ori %expzero, %nan : {vb}
    %result = arith.select %clear, %v0, %inffix : {vb}, {v32}
    %byte = arith.trunci %result : {v32} to {v8}
    vector.maskedstore %y[%jx], %m, %byte : {y}, {vb}, {v8}
   }}
  }}
  return
 }}
}}
''')
    return ''.join(lines)


def emit_wrapper(s: QuantSpec, *, enabled: bool = False) -> str:
    """Preserve generated null, pointer range, and distinct-storage admission."""
    _admit(s, enabled)
    n = s.groups * s.group_elements
    return f'''#include <stddef.h>
#include <stdint.h>
typedef struct {{ void *allocated, *aligned; int64_t offset,size,stride; }} D;
extern void _mlir_ciface_{s.symbol}_leaf(D *,D *,D *);
static int range(const void *p,size_t n) {{return p && n && (uintptr_t)p<=UINTPTR_MAX-n;}}
static int overlap(const void *a,size_t an,const void *b,size_t bn) {{return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an;}}
int {s.symbol}(void *context,const void *const *inputs,void *const *outputs) {{
 (void)context;
 if (!inputs || !outputs) return 1;
 const void *x=inputs[0]; void *y=outputs[0],*e=outputs[1];
 if (!range(x,{2*n}u)||!range(y,{n}u)||!range(e,{4*s.groups}u)||
     overlap(x,{2*n}u,y,{n}u)||overlap(x,{2*n}u,e,{4*s.groups}u)||overlap(y,{n}u,e,{4*s.groups}u)) return 1;
 D dx={{(void *)x,(void *)x,0,{2*n},1}},dy={{y,y,0,{n},1}},de={{e,e,0,{4*s.groups},1}};
 _mlir_ciface_{s.symbol}_leaf(&dx,&dy,&de);
 return 0;
}}
'''


def compile_kernel(spec: QuantSpec, *, work: str | Path, mlir_opt: str | Path, mlir_translate: str | Path,
                   clang: str | Path, enabled: bool = False, timeout: int = 120,
                   march: str = "rv64gcv_zvl256b") -> dict:
    """Compile the opt-in leaf and region wrapper with explicit tools; create a fresh work directory."""
    _admit(spec, enabled)
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    flags = with_march(CODEGEN_FLAGS, march)
    work = Path(work).absolute()
    work.mkdir(parents=True, exist_ok=False)
    src, lowered, ll, leaf_obj, wrapper, wrapper_obj = (work / name for name in (
        "leaf.mlir", "leaf.llvm.mlir", "leaf.ll", "leaf.o", "wrapper.c", "wrapper.o"))
    src.write_text(emit_mlir(spec, enabled=True))
    wrapper.write_text(emit_wrapper(spec, enabled=True))
    commands = [
        [str(mlir_opt), str(src), "--pass-pipeline=builtin.module(" + PIPELINE + ")", "-o", str(lowered)],
        [str(mlir_translate), "--mlir-to-llvmir", str(lowered), "-o", str(ll)],
        [str(clang), *flags, "-c", str(ll), "-o", str(leaf_obj)],
        [str(clang), *flags, "-c", str(wrapper), "-o", str(wrapper_obj)],
    ]
    for index, command in enumerate(commands):
        with (work / f"stage{index}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    return {"source": str(src), "llvm_ir": str(ll), "wrapper": str(wrapper),
            "objects": [str(leaf_obj), str(wrapper_obj)], "symbol": spec.symbol,
            "pipeline": PIPELINE, "commands": commands, "lanes": spec.lanes}
