"""Explicit exact-BF16 provider admission, source and compiler contracts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from test_backend import BACKEND

API = BACKEND.exact_bf16_linear_api()


def spec(**changes):
    return API.LinearSpec(**{"m": 2, "k": 17, "n": 7, **changes})


TOOLS = dict(mlir_opt="selected-opt", mlir_translate="selected-translate", clang="selected-clang")


@pytest.fixture
def compiled_leaf(monkeypatch):
    """Isolate wrapper orchestration from the separately tested MLIR compiler."""
    leaf = API.leaf
    calls = []

    def compile_leaf(selected, **kwargs):
        calls.append((selected, kwargs))
        work = Path(kwargs["work"])
        work.mkdir()
        source, obj = work / "kernel.mlir", work / "kernel.o"
        source.write_text(leaf.emit_mlir(selected))
        obj.write_bytes(b"selected-leaf-object")
        return dict(source=str(source), object=str(obj), commands=[], lanes=4, rows=3)

    monkeypatch.setattr(leaf, "compile_leaf", compile_leaf)
    return calls


def test_default_off_emits_no_source_and_creates_no_artifacts(tmp_path):
    assert API.emit_mlir(spec()) is None
    work = tmp_path / "disabled"
    result = API.compile_kernel(
        spec(), work=work, compiler=tmp_path / "absent-compiler"
    )
    assert not work.exists()
    assert result["enabled"] is False


@pytest.mark.parametrize("enabled", [0, 1, None, "true", []])
def test_opt_in_requires_an_explicit_boolean(tmp_path, enabled):
    with pytest.raises(ValueError):
        API.emit_mlir(spec(), enabled=enabled)
    work = tmp_path / "invalid"
    with pytest.raises(ValueError):
        API.compile_kernel(
            spec(), work=work, compiler=tmp_path / "unused", enabled=enabled
        )
    assert not work.exists()


@pytest.mark.parametrize("dimension", ["m", "k", "n"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", 1 << 63])
def test_invalid_dimensions_fail_closed(dimension, value):
    with pytest.raises(ValueError):
        spec(**{dimension: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"m": 1 << 62, "k": 2},
        {"n": 1 << 62, "k": 2},
        {"m": 1 << 62, "n": 2},
    ],
)
def test_byte_spans_must_fit_signed_rv64(changes):
    with pytest.raises(ValueError):
        spec(**changes)


@pytest.mark.parametrize(
    "symbol",
    ["", "a-b", "1leaf", "å", "x\nreturn", 'x"y',
     "for", "return", "float", "_reserved", "malloc", "memcpy", "size_t",
     "bf", "exact_kernel", "MAC", "MDIMu_SYMBOL", "selected_linear",
     "rvv_exact_bf16_linear_"],
)
def test_symbol_cannot_change_or_inject_source(symbol):
    with pytest.raises(ValueError):
        spec(symbol=symbol)


@pytest.mark.parametrize("suffix", ["bf", "exact_kernel", "MAC", "MDIMu_SYMBOL", "123"])
def test_public_symbol_is_preserved_without_rewriting_its_suffix(suffix):
    symbol = f"rvv_exact_bf16_linear_{suffix}"
    source = API.emit_wrapper(spec(symbol=symbol))
    assert f"int {symbol}(void *context," in source


def test_target_override_preserves_shipped_arithmetic_flags():
    assert API.compiler_flags() == API.CFLAGS
    flags = API.compiler_flags("rv64gcv_zvl256b")
    assert [flag for flag in flags if flag.startswith("-march=")] == [
        "-march=rv64gcv_zvl256b"
    ]
    assert tuple(flag for flag in flags if not flag.startswith("-march=")) == tuple(
        flag for flag in API.CFLAGS if not flag.startswith("-march=")
    )


@pytest.mark.parametrize("march", [None, "", "rv32gcv", "rv64gcv -ffast-math"])
def test_invalid_target_override_refuses(march):
    with pytest.raises(ValueError):
        API.compiler_flags(march)


@pytest.mark.parametrize("timeout", [0, -1, True, 1.5])
def test_invalid_compile_timeout_has_no_side_effects(tmp_path, timeout):
    work = tmp_path / "invalid"
    with pytest.raises(ValueError):
        API.compile_kernel(
            spec(), compiler="unused", work=work, enabled=True, timeout=timeout
        )
    assert not work.exists()


def test_existing_build_directory_is_never_reused(tmp_path):
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError):
        API.compile_kernel(spec(), compiler="unused", work=tmp_path, enabled=True, **TOOLS)
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "unchanged"


@pytest.mark.parametrize("change", [{name: None} for name in TOOLS] + [{"llc": "selected-llc"}])
def test_enabled_compile_requires_each_explicit_lowering_tool(tmp_path, change):
    tools = {**TOOLS, **change}
    work = tmp_path / "missing-tool"
    with pytest.raises(ValueError, match="explicitly"):
        API.compile_kernel(spec(), compiler="unused", work=work, enabled=True, **tools)
    assert not work.exists()


def test_compile_records_lowering_wrapper_flags_and_original_weight_contract(
    tmp_path, monkeypatch, compiled_leaf
):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[command.index("-o") + 1]).write_bytes(b"test-object")

    monkeypatch.setattr(API.subprocess, "run", run)
    compiler = tmp_path / "selected compiler"
    march = "rv64gcv_zvl256b"
    result = API.compile_kernel(
        spec(), compiler=compiler, work=tmp_path / "build", enabled=True,
        timeout=37, march=march, **TOOLS,
    )
    selected, lowering = compiled_leaf[0]
    assert (selected.m, selected.k, selected.n) == (2, 17, 7)
    assert selected.symbol == spec().symbol + "_leaf"
    assert all(lowering[name] == value for name, value in TOOLS.items())
    assert lowering["timeout"] == 37 and lowering["march"] == march
    compile_command = next(command for command, _ in calls if "-c" in command)
    link_command = next(command for command, _ in calls if "-r" in command)
    assert compile_command[0] == link_command[0] == str(compiler)
    assert all(flag in compile_command for flag in API.compiler_flags(march))
    assert result["wrapper"] in compile_command
    assert result["leaf"]["object"] in link_command
    assert link_command[link_command.index("-o") + 1] == result["object"]
    assert all(kwargs["check"] is True and kwargs["timeout"] == 37 for _, kwargs in calls)
    assert result["commands"] == [command for command, _ in calls]
    assert Path(result["source"]).read_text() == API.emit_mlir(spec(), enabled=True)
    assert Path(result["wrapper"]).read_text() == API.emit_wrapper(spec())
    assert result["lanes"] == result["leaf"]["lanes"] == 4
    assert result["rows"] == result["leaf"]["rows"] == 3
    assert json.loads((tmp_path / "build" / "report.json").read_text()) == result


def test_failed_wrapper_compile_retains_source_and_log_without_success_receipt(
    tmp_path, monkeypatch, compiled_leaf
):
    def fail(command, **kwargs):
        kwargs["stdout"].write("selected compiler rejected source\n")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(API.subprocess, "run", fail)
    work = tmp_path / "failed"
    with pytest.raises(subprocess.CalledProcessError):
        API.compile_kernel(
            spec(), compiler="selected", work=work, enabled=True, **TOOLS
        )
    assert (work / "wrapper.c").read_text() == API.emit_wrapper(spec())
    assert (work / "leaf" / "kernel.mlir").read_text() == API.emit_mlir(spec(), enabled=True)
    assert "rejected source" in (work / "wrapper0.log").read_text()
    assert not (work / "report.json").exists()


def test_clang_accepts_portable_wrapper(tmp_path):
    compiler = os.environ.get("RVV_EXACT_CLANG", "clang")
    if not shutil.which(compiler):
        pytest.skip("set RVV_EXACT_CLANG to a RISC-V-capable clang")
    sysroot = os.environ.get("RVV_EXACT_SYSROOT")
    if not sysroot:
        pytest.skip("set RVV_EXACT_SYSROOT to the RISC-V C library sysroot")
    source, obj = tmp_path / "wrapper.c", tmp_path / "wrapper.o"
    source.write_text(API.emit_wrapper(spec()))
    subprocess.run(
        [compiler, "--target=riscv64-unknown-elf", f"--sysroot={sysroot}",
         *API.CFLAGS, "-std=c11", "-Werror", "-c", str(source), "-o", str(obj)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    header = obj.read_bytes()[:20]
    assert header[:6] == b"\x7fELF\x02\x01"
    assert int.from_bytes(header[18:20], "little") == 243  # EM_RISCV
