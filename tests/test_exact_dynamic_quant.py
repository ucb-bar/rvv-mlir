"""Integer BF16 quantization equality, tail, invalid ABI, and byte alignment checks."""
import ctypes
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rvv.transforms import exact_dynamic_quant as q


def gold(words, size):
    codes, exponents = bytearray(), bytearray()
    for start in range(0, len(words), size):
        block = words[start:start+size]
        maximum = max([b & 32767 for b in block if b & 32640 != 32640] + [0])
        exponent = (maximum >> 7)-135 if maximum >> 7 else maximum.bit_length()-142 if maximum else 0
        exponents += struct.pack('<i', exponent)
        shift = max(-127, min(127, exponent))
        for b in block:
            sign, e, mant = b >> 15, (b >> 7) & 255, b & 127
            sat = 254 if sign else 126
            trunc, tail = (128 | mant) >> 4, (128 | mant) & 15
            rounded = trunc + (tail > 8 or (tail == 8 and trunc & 1))
            carry = rounded == 16
            final = e-127-shift+carry
            code = sign << 7 | ((final+7) & 15) << 3 | (0 if carry else (rounded-8) & 7)
            if code & 127 == 127: code = sat
            if final > 8: code = sat
            elif final < -6: code = 0
            if e == 255 and not mant: code = sat
            if not e or (e == 255 and mant): code = 0
            codes.append(code)
    return bytes(codes), bytes(exponents)


def tools():
    opt = os.environ.get('RVV_MLIR_OPT') or shutil.which('mlir-opt')
    translate = os.environ.get('RVV_MLIR_TRANSLATE') or shutil.which('mlir-translate')
    clang = os.environ.get('RVV_EXACT_CLANG') or shutil.which('clang')
    if not all((opt,translate,clang)): pytest.skip('select explicit MLIR and Clang tools')
    return opt, translate, clang


def build(tmp, spec):
    opt, translate, clang = tools()
    tmp.mkdir(exist_ok=True)
    (tmp/'leaf.mlir').write_text(q.emit_mlir(spec,enabled=True))
    (tmp/'wrapper.c').write_text(q.emit_wrapper(spec,enabled=True))
    commands = [[opt,tmp/'leaf.mlir','--pass-pipeline=builtin.module('+q.PIPELINE+')','-o',tmp/'leaf.llvm.mlir'],
                [translate,'--mlir-to-llvmir',tmp/'leaf.llvm.mlir','-o',tmp/'leaf.ll'],
                [clang,'-shared','-fPIC','-O2',tmp/'leaf.ll',tmp/'wrapper.c','-o',tmp/'leaf.so']]
    for i, cmd in enumerate(commands):
        with (tmp/f'build-{i}.log').open('w') as log:
            subprocess.run(list(map(str,cmd)),check=True,stdout=log,stderr=subprocess.STDOUT)
    lib = ctypes.CDLL(str(tmp/'leaf.so'))
    f = getattr(lib,spec.symbol)
    f.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_void_p)]
    f.restype = ctypes.c_int
    return lib, f


@pytest.mark.parametrize('kw',[{'groups':0},{'group_elements':True},{'lanes':3},{'lanes':64},{'symbol':'bad-name'},{'symbol':'range'},{'symbol':'overlap'},{'symbol':'D'},{'symbol':'int'},{'symbol':'rvv_exact_dynamic_quant_'},{'symbol':7},{'groups':1<<62}])
def test_refuse(kw):
    with pytest.raises(ValueError): q.QuantSpec(**({'groups':2,'group_elements':17}|kw))


def test_explicit_opt_in(tmp_path):
    spec = q.QuantSpec(2,17)
    with pytest.raises(ValueError): q.emit_mlir(spec)
    with pytest.raises(ValueError): q.emit_wrapper(spec,enabled=1)
    with pytest.raises(TypeError): q.emit_mlir(None,enabled=True)
    with pytest.raises(ValueError):
        q.compile_kernel(spec,work=tmp_path/'off',mlir_opt='unused',mlir_translate='unused',clang='unused')
    assert not (tmp_path/'off').exists()


def test_compile_kernel_emits_riscv_vector_objects(tmp_path):
    opt, translate, clang = tools()
    result = q.compile_kernel(q.QuantSpec(2,17,symbol='rvv_exact_dynamic_quant_custom'),work=tmp_path/'kernel',mlir_opt=opt,mlir_translate=translate,clang=clang,enabled=True)
    for path in result['objects']:
        data = Path(path).read_bytes()
        assert data[:6] == b'\x7fELF\x02\x01' and int.from_bytes(data[18:20],'little') == 243
        assert b'zvl256b' in data
    assert '-march=rv64gcv_zvl256b' in result['commands'][-1]
    assert result['symbol'] == 'rvv_exact_dynamic_quant_custom'


@pytest.mark.parametrize('size',[1,7,8,9,17,257,65536])
def test_exact_tails_unaligned_and_abi(tmp_path,size):
    groups = 1 if size == 65536 else 5
    spec = q.QuantSpec(groups,size)
    lib,f = build(tmp_path,spec)
    rng = random.Random(731)
    words = list(range(65536)) if size == 65536 else [rng.randrange(65536) for _ in range(groups*size)]
    if size != 65536:
        words[:size] = [0]*size
        words[size:2*size] = [1+(i%127) for i in range(size)]
        words[2*size:3*size] = [0x7f80+(i%128) for i in range(size)]
    raw = struct.pack('<'+'H'*len(words),*words)
    x = ctypes.create_string_buffer(b'X'+raw+b'Y')
    ys = [ctypes.create_string_buffer(b'G'*(n+2)) for n in (len(words),4*groups)]
    inputs = (ctypes.c_void_p*1)(ctypes.addressof(x)+1)
    outputs = (ctypes.c_void_p*2)(*[ctypes.addressof(y)+1 for y in ys])
    assert f(None,inputs,outputs) == 0
    expected = gold(words,size)
    for buf, want in zip(ys,expected):
        assert buf.raw[1:len(want)+1] == want
        assert buf.raw[0:1] == b'G' and buf.raw[len(want)+1:len(want)+2] == b'G'
    assert x.raw[:len(raw)+2] == b'X'+raw+b'Y'
    assert f(None,None,outputs) == 1
    assert f(None,inputs,None) == 1
    badoutputs = (ctypes.c_void_p*2)(inputs[0],outputs[1])
    assert f(None,inputs,badoutputs) == 1
    assert f(None,inputs,(ctypes.c_void_p*2)(outputs[0],inputs[0])) == 1
    assert f(None,inputs,(ctypes.c_void_p*2)(outputs[0],outputs[0])) == 1
    assert f(None,(ctypes.c_void_p*1)(ctypes.c_void_p(-1)),outputs) == 1


@pytest.mark.parametrize('anchor',[0,1,127,128,0x3f80,0x7f7f])
def test_all_bf16_encodings_with_group_maxima(tmp_path,anchor):
    """Each BF16 encoding is paired with six controlled maximum anchors."""
    spec=q.QuantSpec(65536,2)
    lib,f=build(tmp_path,spec)
    words=[v for b in range(65536) for v in (b,anchor)]
    raw=struct.pack('<'+'H'*len(words),*words)
    x=ctypes.create_string_buffer(raw)
    expected=gold(words,2)
    ys=[ctypes.create_string_buffer(len(e)) for e in expected]
    inputs=(ctypes.c_void_p*1)(ctypes.addressof(x))
    outputs=(ctypes.c_void_p*2)(*[ctypes.addressof(y) for y in ys])
    assert f(None,inputs,outputs)==0
    assert all(y.raw==e for y,e in zip(ys,expected))


def test_prefill64_geometry(tmp_path):
    spec=q.QuantSpec(32,4096)
    lib,f=build(tmp_path,spec)
    words=[((i*731)^0x9e37)&65535 for i in range(32*4096)]
    x=ctypes.create_string_buffer(struct.pack('<'+'H'*len(words),*words))
    expected=gold(words,4096)
    ys=[ctypes.create_string_buffer(len(e)) for e in expected]
    ins=(ctypes.c_void_p*1)(ctypes.addressof(x))
    outs=(ctypes.c_void_p*2)(*[ctypes.addressof(y) for y in ys])
    assert f(None,ins,outs)==0
    assert all(y.raw==e for y,e in zip(ys,expected))
