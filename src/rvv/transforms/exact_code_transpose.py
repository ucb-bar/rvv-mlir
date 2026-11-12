"""Default-off integer byte transpose over caller-owned buffers.

Transpose maps [groups,rows,columns] to [groups,columns,rows].
The ordinary provider pipeline lowers vector.gather to RVV strided loads.
The wrapper retains the generated layout emitter's null, range, and alias checks.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

from rvv.transforms.exact_bf16_leaf import PIPELINE


@dataclass(frozen=True)
class CodeTransposeSpec:
    groups: int
    rows: int
    columns: int
    lanes: int = 16
    symbol: str = "rvv_code_transpose"

    def __post_init__(self):
        for name in ("groups", "rows", "columns", "lanes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if self.lanes > 64 or self.lanes & (self.lanes - 1):
            raise ValueError("lanes must be a power of two no greater than 64")
        if self.groups * self.rows * self.columns > 2**63 - 1:
            raise ValueError("buffer span exceeds signed index range")
        if (self.groups * self.rows * self.columns + (self.lanes - 1) * self.columns > 2**63 - 1
                or self.rows + self.lanes - 1 > 2**63 - 1):
            raise ValueError("gather offsets exceed signed index range")
        if not isinstance(self.symbol, str) or not re.fullmatch(r"rvv_code_transpose(?:_[A-Za-z0-9_]+)?", self.symbol):
            raise ValueError("symbol must use the rvv_code_transpose namespace")

    @property
    def bytes(self):
        return self.groups * self.rows * self.columns


def _admit(spec, enabled):
    if enabled is not True:
        raise ValueError("exact code transpose lowering requires enabled=True")
    if not isinstance(spec, CodeTransposeSpec):
        raise TypeError("CodeTransposeSpec is required")

def emit_mlir(spec, *, enabled=False):
    _admit(spec, enabled)
    symbol = spec.symbol
    L = spec.lanes
    T = f"memref<{spec.bytes}xi8, strided<[1], offset: ?>>"
    VI, VB, V = (f"vector<{L}x{kind}>" for kind in ("index", "i1", "i8"))
    return f'''module {{
     func.func @{symbol}_leaf(%x: {T}, %y: {T}) attributes {{llvm.emit_c_interface}} {{
      %z = arith.constant 0 : index
      %one = arith.constant 1 : index
      %groups = arith.constant {spec.groups} : index
      %columns = arith.constant {spec.columns} : index
      %rows = arith.constant {spec.rows} : index
      %groupstride = arith.constant {spec.rows * spec.columns} : index
      %lanes = arith.constant {L} : index
      %lane = arith.constant dense<[{', '.join(str(i*spec.columns) for i in range(L))}]> : {VI}
      %zero = arith.constant dense<0> : {V}
      scf.for %g = %z to %groups step %one {{
       %base = arith.muli %g, %groupstride : index
       scf.for %c = %z to %columns step %one {{
        %source = arith.addi %base, %c : index
        %column = arith.muli %c, %rows : index
        %dest = arith.addi %base, %column : index
        scf.for %r = %z to %rows step %lanes {{
         %left = arith.subi %rows, %r : index
         %mask = vector.create_mask %left : {VB}
         %offset = arith.muli %r, %columns : index
         %at = arith.addi %source, %offset : index
         %value = vector.gather %x[%at][%lane], %mask, %zero : {T}, {VI}, {VB}, {V} into {V}
         %outat = arith.addi %dest, %r : index
         vector.maskedstore %y[%outat], %mask, %value {{alignment = 1 : i64}} : {T}, {VB}, {V}
        }}
       }}
      }}
      return
     }}
    }}
    '''

def emit_wrapper(spec, *, enabled=False):
    _admit(spec, enabled)
    n, symbol = spec.bytes, spec.symbol
    return f'''#include <stddef.h>
#include <stdint.h>
_Static_assert(SIZE_MAX >= {n}ULL,"transpose span exceeds size_t");
typedef struct {{void *allocated,*aligned;int64_t offset,size,stride;}} D;
extern void _mlir_ciface_{symbol}_leaf(D*,D*);
int {symbol}(void *context,const void *const *inputs,void *const *outputs) {{
 (void)context;
 if (!inputs||!outputs||!outputs[0]) return -1;
 uintptr_t b=(uintptr_t)outputs[0];
 if (!inputs[0]||b>UINTPTR_MAX-{n}u||(uintptr_t)inputs[0]>UINTPTR_MAX-{n}u||((uintptr_t)inputs[0]<b+{n}u&&b<(uintptr_t)inputs[0]+{n}u)) return -1;
 D x={{(void*)inputs[0],(void*)inputs[0],0,{n},1}},y={{outputs[0],outputs[0],0,{n},1}};
 _mlir_ciface_{symbol}_leaf(&x,&y);return 0;
}}
'''
