"""Generic byte transpose geometry, tails, guards, and failure behavior."""
import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rvv.transforms import exact_code_transpose as api

@pytest.mark.parametrize('changes',[{'groups':0},{'rows':True},{'columns':-1},{'lanes':3},{'lanes':128},{'groups':1<<62},{'columns':1<<62},{'symbol':'x;bad'},{'symbol':'D'},{'symbol':'int'},{'symbol':'rvv_code_transpose_'},{'symbol':1}])
def test_invalid_geometry(changes):
    with pytest.raises(ValueError):api.CodeTransposeSpec(**({'groups':2,'rows':3,'columns':5}|changes))

def test_explicit_opt_in():
    spec=api.CodeTransposeSpec(2,3,5)
    with pytest.raises(ValueError):api.emit_mlir(spec)
    with pytest.raises(ValueError):api.emit_wrapper(spec,enabled=1)
    with pytest.raises(TypeError):api.emit_mlir(None,enabled=True)

@pytest.mark.parametrize('geometry',[(1,1,1,1),(2,3,5,4),(3,17,7,8),(2,16,19,16),(1,65,64,16),(2,31,3,32)])
def test_native_mapping_tails_guards_and_refusals(tmp_path,geometry):
    tools=[os.environ.get(env)or shutil.which(name)for env,name in [('RVV_MLIR_OPT','mlir-opt'),('RVV_MLIR_TRANSLATE','mlir-translate'),('RVV_EXACT_CLANG','clang')]]
    if not all(tools):pytest.skip('select explicit MLIR and Clang tools')
    spec=api.CodeTransposeSpec(*geometry,symbol='rvv_code_transpose'+('_custom' if geometry[3]>8 else ''))
    (tmp_path/'leaf.mlir').write_text(api.emit_mlir(spec,enabled=True))
    (tmp_path/'wrapper.c').write_text(api.emit_wrapper(spec,enabled=True))
    opt,translate,clang=tools
    commands=[[opt,tmp_path/'leaf.mlir','--pass-pipeline=builtin.module('+api.PIPELINE+')','-o',tmp_path/'leaf.llvm.mlir'],[translate,'--mlir-to-llvmir',tmp_path/'leaf.llvm.mlir','-o',tmp_path/'leaf.ll'],[clang,'-O2','-fPIC','-shared',tmp_path/'leaf.ll',tmp_path/'wrapper.c','-o',tmp_path/'leaf.so']]
    for i,cmd in enumerate(commands):
        with(tmp_path/f'{i}.log').open('w')as f:subprocess.run(list(map(str,cmd)),check=True,stdout=f,stderr=subprocess.STDOUT,timeout=120)
    library=ct.CDLL(str(tmp_path/'leaf.so'));function=getattr(library,spec.symbol)
    function.restype=ct.c_int;function.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_void_p)]
    raw=bytes((i*73+i//7)%256 for i in range(spec.bytes))
    expected=bytes(raw[(g*spec.rows+r)*spec.columns+c]for g in range(spec.groups)for c in range(spec.columns)for r in range(spec.rows))
    source=ct.create_string_buffer(b'X'+raw+b'Y');target=ct.create_string_buffer(b'G'*(spec.bytes+2))
    inputs=(ct.c_void_p*1)(ct.addressof(source)+1);outputs=(ct.c_void_p*1)(ct.addressof(target)+1)
    assert function(None,inputs,outputs)==0
    assert target.raw[:spec.bytes+2]==b'G'+expected+b'G'
    assert source.raw[:spec.bytes+2]==b'X'+raw+b'Y'
    before=target.raw
    assert function(None,None,outputs)==-1
    assert function(None,inputs,None)==-1
    assert function(None,inputs,(ct.c_void_p*1)(0))==-1
    assert function(None,(ct.c_void_p*1)(0),outputs)==-1
    assert function(None,inputs,inputs)==-1
    assert function(None,(ct.c_void_p*1)((1<<64)-spec.bytes+1),outputs)==-1
    assert target.raw==before
