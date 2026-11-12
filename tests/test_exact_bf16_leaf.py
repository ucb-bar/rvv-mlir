"""MLIR-authored leaf validation; native host equality uses explicitly selected tools."""
from __future__ import annotations

import ctypes
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rvv.transforms import exact_bf16_leaf as API


@pytest.mark.parametrize("changes", [{"m": 0}, {"k": True}, {"n": 1 << 63},
                                         {"target": API.PanelTarget(128,64,8,32,1,1024)}, {"symbol": "a;bad"},
                                         {"m": 1 << 62, "k": 2}, {"weight_layout": "NN"}])
def test_invalid_specs_fail_closed(changes):
    with pytest.raises(ValueError):
        API.LeafSpec(**({"m": 2, "k": 9, "n": 7} | changes))


def _tools():
    selected = [os.environ.get(name) or shutil.which(default) for name, default in (
        ("RVV_MLIR_OPT", "mlir-opt"), ("RVV_MLIR_TRANSLATE", "mlir-translate"),
        ("RVV_EXACT_CLANG", "clang"),
    )]
    if not all(selected):
        pytest.skip("select RVV_MLIR_OPT, RVV_MLIR_TRANSLATE and RVV_EXACT_CLANG")
    return selected


class Descriptor(ctypes.Structure):
    _fields_ = [("allocated", ctypes.POINTER(ctypes.c_uint16)),
                ("aligned", ctypes.POINTER(ctypes.c_uint16)),
                ("offset", ctypes.c_int64), ("size", ctypes.c_int64),
                ("stride", ctypes.c_int64)]


def _bf_float(x):
    return struct.unpack("<f", struct.pack("<I", x << 16))[0]


def _f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return float("-inf") if x < 0 else float("inf")


def _gold(a, w, m, k, n):
    result = []
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for kk in range(k):
                acc = _f32(acc + _f32(_bf_float(a[i*k+kk]) * _bf_float(w[j*k+kk])))
            bits = struct.unpack("<I", struct.pack("<f", acc))[0]
            result.append(((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffffffff) >> 16)
    return result


@pytest.mark.parametrize("m", [2, 5, 8])
@pytest.mark.parametrize("n", [7, 11])
@pytest.mark.parametrize("layout", ["NK", "KN"])
def test_generated_host_equality_order_tails_offsets_and_overflow(tmp_path, m, n, layout):
    opt, translate, clang = _tools()
    spec = API.LeafSpec(m, 9, n, weight_layout=layout)
    source, lowered, llvm, shared = (tmp_path / n for n in ("leaf.mlir", "leaf.llvm.mlir", "leaf.ll", "leaf.so"))
    source.write_text(API.emit_mlir(spec))
    commands = [[opt, str(source), "--pass-pipeline=builtin.module("+API.PIPELINE+")", "-o", str(lowered)],
                [translate, "--mlir-to-llvmir", str(lowered), "-o", str(llvm)]]
    commands.append([clang, "-O2", "-shared", "-fPIC", "-ffp-contract=off", "-fno-fast-math", str(llvm), "-o", str(shared)])
    for stage, cmd in enumerate(commands):
        with (tmp_path / f"stage{stage}.log").open("w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    lib = ctypes.CDLL(str(shared))
    leaf = getattr(lib, "_mlir_ciface_"+spec.symbol)
    leaf.argtypes = [ctypes.POINTER(Descriptor)]*3
    leaf.restype = ctypes.c_int32
    corpus = [0x0000, 0x8000, 0x0001, 0x8001, 0x3f80, 0xbf80, 0x3f81,
              0x4000, 0xc000, 0x3eab, 0x3f00, 0x3380]
    a = [corpus[(i*5+3) % len(corpus)] for i in range(m*9)]
    w = [corpus[(i*7+5) % len(corpus)] for i in range(n*9)]
    def invoke(av, wv):
        if layout == "KN":
            wv = [wv[j*9+k] for k in range(9) for j in range(n)]
        arrays = [(ctypes.c_uint16*(len(v)+4))(*([0x5a5a]+v+[0x5a5a]*3)) for v in (av,wv,[0x5a5a]*(m*n))]
        descs = [Descriptor(x,x,1,len(x)-4,1) for x in arrays]
        status = leaf(*(ctypes.byref(d) for d in descs))
        assert list(arrays[0]) == [0x5a5a]+av+[0x5a5a]*3
        assert list(arrays[1]) == [0x5a5a]+wv+[0x5a5a]*3
        assert arrays[2][0] == 0x5a5a and list(arrays[2])[-3:] == [0x5a5a]*3
        return status,list(arrays[2])[1:1+m*n]
    status, values = invoke(a,w)
    assert status == 0
    assert values == _gold(a,w,m,9,n)
    # Ascending F32 adds lose the middle 1; reassociation would produce 1.
    ordered_a = [0x4b80, 0x3f80, 0xcb80] + [0]*6
    status, values = invoke(ordered_a*m, [0x3f80]*(n*9))
    assert status == 0 and values == [0]*(m*n)
    # F32 stays finite, but a BF16 halfway rounding overflows.
    rounding_overflow = [0x7f7f, 0x7b00] + [0]*7
    status, _ = invoke(rounding_overflow*m, [0x3f80]*(n*9))
    assert status == 1
    # F32 overflow must also be reported to the transactional wrapper.
    status, _ = invoke([0x7f7f]*(m*9), [0x3f80]*(n*9))
    assert status == 1


@pytest.mark.parametrize("vlen,lanes", [(128,8), (256,16), (512,32)])
def test_policy_geometry_and_shared_weight_load(vlen, lanes):
    target = API.PanelTarget(vlen,64,16,32,1,65536)
    spec = API.LeafSpec(5,37,45,target=target)
    assert spec.lanes == lanes
    assert spec.geometry.mr == 4
    source = API.emit_mlir(spec)
    full, tail = source.split('%tailbad =')
    assert full.count('vector.gather') == 1
    assert full.count('arith.mulf') == full.count('arith.addf') == 4
    assert tail.count('vector.gather') == tail.count('arith.mulf') == tail.count('arith.addf') == 1
    assert 'contract' not in source and 'fastmath' not in source


def test_register_budget_limits_rows_and_workspace_limits_lanes():
    target = API.PanelTarget(256,64,16,32,1,4*37*3,register_budget=12)
    spec = API.LeafSpec(5,37,45,target=target)
    assert spec.geometry.mr == 1
    assert spec.lanes == 3


@pytest.mark.parametrize("target_bits,march", [
    (256, "rv64gcv"), (512, "rv64gcv_zvl256b"),
    (128, "rv64gcv_zvl64b"), (128, "rv64gcv_zvl384b"),
    (128, "rv64gcv_zvl128b_zvl256b"),
])
def test_target_and_codegen_minimum_vlen_must_agree(tmp_path, target_bits, march):
    target = API.PanelTarget(target_bits, 64, 16, 32, 1, 4096)
    with pytest.raises(ValueError, match="VLEN"):
        API.compile_leaf(API.LeafSpec(4, 9, 17, target=target), work=tmp_path / "leaf",
                         mlir_opt="unused", mlir_translate="unused", clang="unused",
                         march=march)
    assert not (tmp_path / "leaf").exists()


@pytest.mark.parametrize("vlen", [128, 256, 512])
def test_production_clang_compiles_mlir_leaf_to_riscv_object(tmp_path, vlen):
    opt, translate, clang = _tools()
    target = API.PanelTarget(vlen, 64, 16, 32, 1, 4096)
    result = API.compile_leaf(
        API.LeafSpec(5, 9, 17, target=target), work=tmp_path / "leaf",
        mlir_opt=opt, mlir_translate=translate, clang=clang,
        march=f"rv64gcv_zvl{vlen}b",
    )
    data = Path(result["object"]).read_bytes()
    assert data[:6] == b"\x7fELF\x02\x01"
    assert int.from_bytes(data[18:20], "little") == 243
    assert int.from_bytes(data[48:52], "little") & 6 == 4  # Double-float ABI.
    assert f"zvl{vlen}b".encode() in data
    assert "-ffp-contract=off" in result["commands"][-1]
    assert result["contraction"] is False


def test_llc_uses_selected_vlen_and_disables_contraction(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        Path(command[command.index("-o") + 1]).write_text("; mock lowering\n")

    monkeypatch.setattr(API.subprocess, "run", run)
    target = API.PanelTarget(512, 64, 16, 32, 1, 4096)
    API.compile_leaf(API.LeafSpec(4, 9, 17, target=target), work=tmp_path / "leaf",
                     mlir_opt="opt", mlir_translate="translate", llc="llc",
                     march="rv64gcv_zvl512b")
    assert "-mattr=+m,+a,+f,+d,+c,+v,+zvl512b" in calls[-1]
    assert "-fp-contract=off" in calls[-1]
