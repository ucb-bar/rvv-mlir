"""Explicit host/native full-storage proof for borrowed I8/I32 compiler leaves.

Tools and Merlin are explicit inputs. Results require a fresh output directory.
This is a generic one-hart RVV compiler unit proof, not a chip configuration,
performance benchmark, or transport of all target output values.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(arguments, output, name, timeout=120):
    with (
        (output / (name + ".stdout")).open("wb") as stdout,
        (output / (name + ".stderr")).open("wb") as stderr,
    ):
        process = subprocess.Popen(
            list(map(str, arguments)),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise RuntimeError(name + " exceeded its owned timeout") from None
    if status:
        raise RuntimeError(name + " failed; inspect its preserved stderr")


class Descriptor(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
    ]


def descriptor(array, offset, sizes, strides):
    return Descriptor(
        array.ctypes.data,
        array.ctypes.data,
        offset,
        (ctypes.c_int64 * 2)(*sizes),
        (ctypes.c_int64 * 2)(*strides),
    )


def inputs(a_pitch, b_pitch, c_pitch, trial):
    """Eight independent patterns include every I8 value and I32 wrapping."""
    a = ((np.arange(17 * a_pitch, dtype=np.uint64) + min(trial, 5) * 51) & 255).astype(
        np.uint8
    )
    b = ((np.arange(17 * b_pitch, dtype=np.uint64) * 37 + trial * 19) & 255).astype(
        np.uint8
    )
    if trial >= 6:
        a[:] = b[:] = 128 if trial == 6 else 127
    seed = np.full(3 * c_pitch, 0x5A5A5A5A, dtype="<u4")
    for row in range(3):
        for column in range(65):
            seed[row * c_pitch + column] = (
                0x7FFFFFFF if (row + column) & 1 else 0xFFFFFFF0
            )
    av = a.view(np.int8).reshape(17, a_pitch)[:, :3].T.astype(np.int64)
    bv = b.view(np.int8).reshape(17, b_pitch)[:, :65].astype(np.int64)
    expected = seed.copy()
    expected.reshape(3, c_pitch)[:, :65] = (
        seed.reshape(3, c_pitch)[:, :65].astype(np.int64) + av @ bv
    ).astype("<u4")
    return a, b, seed, expected


def literal(values, word=False):
    return ",".join(hex(int(value)) + ("u" if word else "") for value in values)


def driver(fixtures, vlen):
    declarations = [
        "#include <stdint.h>",
        "#include <stddef.h>",
        "typedef struct{void*allocated,*aligned;int64_t offset,size[2],stride[2];} D;",
    ]
    bodies = []
    for number, fixture in enumerate(fixtures):
        ap, bp, cp = fixture["pitches"]
        leaves = fixture["leaves"]
        for leaf in leaves:
            declarations.append(f"extern void _mlir_ciface_{leaf['symbol']}(D*,D*,D*);")
        for trial in range(8):
            a, b, seed, expected = inputs(ap, bp, cp, trial)
            for name, values, word in (
                ("a", a, False),
                ("b", b, False),
                ("seed", seed, True),
                ("gold", expected, True),
            ):
                ctype = "uint32_t" if word else "uint8_t"
                declarations.append(
                    f"static const {ctype} f{number}_{name}_{trial}[]={{"
                    + literal(values, word)
                    + "};"
                )
        declarations.extend(
            [
                f"static struct{{uint8_t lo[64],data[{17 * ap}],hi[64];}} a{number};",
                f"static struct{{uint8_t lo[64],data[{17 * bp}],hi[64];}} b{number};",
                f"static struct{{uint32_t lo[16],data[{3 * cp}],hi[16];}} c{number};",
            ]
        )
        for trial in range(8):
            body = ["{"]
            for name, length in (("a", 17 * ap), ("b", 17 * bp)):
                body.append(
                    f"for(size_t i=0;i<{length};i++){name}{number}.data[i]=f{number}_{name}_{trial}[i];"
                )
                body.append(
                    f"for(size_t i=0;i<64;i++){name}{number}.lo[i]={name}{number}.hi[i]=0x5a;"
                )
            body.append(
                f"for(size_t i=0;i<{3 * cp};i++)c{number}.data[i]=f{number}_seed_{trial}[i];"
            )
            body.append(
                f"for(size_t i=0;i<16;i++)c{number}.lo[i]=c{number}.hi[i]=0x5a5a5a5a;"
            )
            for leaf in leaves:
                m, n, i, j = (leaf[key] for key in ("m", "n", "i", "j"))
                body.extend(
                    [
                        "{",
                        f"D a={{a{number}.data,a{number}.data,{i},{{{m},17}},{{1,{ap}}}}};",
                        f"D b={{b{number}.data,b{number}.data,{j},{{17,{n}}},{{{bp},1}}}};",
                        f"D c={{c{number}.data,c{number}.data,{i * cp + j},{{{m},{n}}},{{{cp},1}}}};",
                        f"_mlir_ciface_{leaf['symbol']}(&a,&b,&c);",
                    ]
                )
                if n > 1:
                    body.append(
                        'unsigned long vl;__asm__ volatile("csrr %0,vl":"=r"(vl)::"memory");'
                        f"if(vl!={leaf['nr']})return 2;"
                    )
                body.append("}")
            for name, length in (("a", 17 * ap), ("b", 17 * bp)):
                body.append(
                    f"for(size_t i=0;i<{length};i++)if({name}{number}.data[i]!=f{number}_{name}_{trial}[i])return 3;"
                )
                body.append(
                    f"for(size_t i=0;i<64;i++)if({name}{number}.lo[i]!=0x5a||{name}{number}.hi[i]!=0x5a)return 4;"
                )
            body.append(
                f"for(size_t i=0;i<{3 * cp};i++)if(c{number}.data[i]!=f{number}_gold_{trial}[i])return 5;"
            )
            body.append(
                f"for(size_t i=0;i<16;i++)if(c{number}.lo[i]!=0x5a5a5a5a||c{number}.hi[i]!=0x5a5a5a5a)return 6;"
            )
            body.append("}")
            bodies.extend(body)
    return "\n".join(declarations) + (
        '\nint main(void){unsigned long vb;__asm__ volatile("csrr %0,vlenb":"=r"(vb));'
        f"if(vb!={vlen // 8})return 1;\n" + "\n".join(bodies) + "\nreturn 0;}\n"
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--merlin-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, default=ROOT)
    parser.add_argument("--compiler-python", type=Path, required=True)
    parser.add_argument("--clang", type=Path, required=True)
    parser.add_argument("--linker", type=Path, required=True)
    parser.add_argument("--objdump", type=Path, required=True)
    parser.add_argument("--dtc", type=Path, required=True)
    parser.add_argument("--spike", type=Path, required=True)
    parser.add_argument("--spike-option", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(ROOT / "src"), str(args.merlin_root / "src")]
    os.environ.update(
        MERLIN_COMPILER_PYTHON=str(args.compiler_python),
        MERLIN_TARGET_PATH=str(args.pipeline_root),
        PYTHONDONTWRITEBYTECODE="1",
        PATH=str(args.dtc.parent) + os.pathsep + os.environ.get("PATH", ""),
    )
    from merlin.llvmlower.lower import lower_model
    from prove_im2col_stream import LINK, START

    from rvv.packing.borrowed_i8_leaf import LeafSpec, emit

    tools = {
        str(path): sha(path)
        for path in (args.clang, args.linker, args.objdump, args.dtc, args.spike)
    }
    report = {
        "schema": "rvv.borrowed_i8_compiler_proof.v1",
        "pass": False,
        "tools": tools,
        "cases": [],
        "pipeline_root": str(args.pipeline_root),
        "builder_root": str(ROOT),
        "native_extra_options": args.spike_option,
        "hardware_contacted": False,
        "scope": "generic one-hart RVV compiler unit; no chip/image or performance qualification",
        "native_method": "guest compares every storage word/input/guard against independent embedded NumPy goldens; not raw value transport",
        "implicit_header_library_closure_complete": False,
    }
    try:
        for vlen in (128, 256, 512, 1024):
            output = args.output / f"vlen{vlen}"
            output.mkdir()
            fixtures = []
            objects = []
            llvm = []
            for number, pitches in enumerate(((3, 65, 65), (5, 68, 69))):
                ap, bp, cp = pitches
                fixture = {"pitches": pitches, "leaves": []}
                for i, m in ((0, 2), (2, 1)):
                    for j, n in ((0, 64), (64, 1)):
                        symbol = f"leaf_f{number}_m{m}_n{n}"
                        emission = emit(
                            symbol,
                            LeafSpec(m, n, 17, ap, bp, cp, vlen),
                            allow_scalar_1x1=True,
                        )
                        work = output / symbol
                        work.mkdir()
                        (work / "source.mlir").write_text(emission.source)
                        lowered = lower_model(
                            emission.source,
                            work / "lower",
                            targets=(),
                            textual=True,
                            codegen_target="rvv",
                            **emission.lowering_kwargs(),
                        )
                        ll = Path(lowered.ll_path)
                        text = ll.read_text()
                        for forbidden in (
                            "@malloc",
                            "@memrefCopy",
                            "@memcpy",
                            "@memmove",
                            " alloca ",
                        ):
                            if forbidden in text:
                                raise RuntimeError(
                                    "borrowed output allocated/copied: " + forbidden
                                )
                        flags = [
                            "--target=riscv64-unknown-elf",
                            f"-march=rv64gcv_zvl{vlen}b",
                            "-mabi=lp64d",
                            "-mcmodel=medany",
                            "-O2",
                            "-ffreestanding",
                            "-fno-builtin",
                            "-fno-vectorize",
                            "-fno-slp-vectorize",
                        ]
                        obj = work / "leaf.o"
                        run(
                            [args.clang, *flags, "-c", ll, "-o", obj],
                            work,
                            "target-compile",
                        )
                        assembly = subprocess.check_output(
                            [args.objdump, "-d", obj], text=True
                        )
                        (work / "leaf.s").write_text(assembly)
                        if any(
                            token in assembly
                            for token in (
                                "vs1r.v",
                                "vs2r.v",
                                "vs4r.v",
                                "vs8r.v",
                                "vl1re",
                                "vl2re",
                                "vl4re",
                                "vl8re",
                            )
                        ):
                            raise RuntimeError("vector register spill")
                        leaf = {
                            "symbol": symbol,
                            "m": m,
                            "n": n,
                            "i": i,
                            "j": j,
                            "nr": emission.nr,
                            "record": emission.record(),
                            "LLVM_sha256": sha(ll),
                            "object_sha256": sha(obj),
                            "vector_MAC_machine_present": "vwmacc.vx" in assembly,
                        }
                        fixture["leaves"].append(leaf)
                        llvm.append(ll)
                        objects.append(obj)
                fixtures.append(fixture)
            library = output / "host.so"
            run(
                [args.clang, "-O2", "-shared", "-fPIC", *llvm, "-o", library],
                output,
                "host-compile",
            )
            loaded = ctypes.CDLL(str(library))
            for fixture in fixtures:
                ap, bp, cp = fixture["pitches"]
                for trial in range(8):
                    a, b, seed, expected = inputs(ap, bp, cp, trial)
                    raw_a = np.full(a.size + 128, 0x5A, np.uint8)
                    raw_b = np.full(b.size + 128, 0x5A, np.uint8)
                    raw_c = np.full(seed.size * 4 + 128, 0x5A, np.uint8)
                    av, bv, cv = raw_a[64:-64], raw_b[64:-64], raw_c[64:-64].view("<u4")
                    av[:] = a
                    bv[:] = b
                    cv[:] = seed
                    original_a, original_b = raw_a.copy(), raw_b.copy()
                    for leaf in fixture["leaves"]:
                        m, n, i, j = (leaf[key] for key in ("m", "n", "i", "j"))
                        descriptors = (
                            descriptor(av, i, (m, 17), (1, ap)),
                            descriptor(bv, j, (17, n), (bp, 1)),
                            descriptor(cv, i * cp + j, (m, n), (cp, 1)),
                        )
                        function = getattr(loaded, "_mlir_ciface_" + leaf["symbol"])
                        function.argtypes = [ctypes.POINTER(Descriptor)] * 3
                        function(*(ctypes.byref(value) for value in descriptors))
                    if not (
                        np.array_equal(cv, expected)
                        and np.array_equal(raw_a, original_a)
                        and np.array_equal(raw_b, original_b)
                        and np.all(raw_c[:64] == 0x5A)
                        and np.all(raw_c[-64:] == 0x5A)
                    ):
                        raise RuntimeError("host full-storage/input/guard mismatch")
            (output / "main.c").write_text(driver(fixtures, vlen))
            (output / "start.S").write_text(START)
            (output / "link.ld").write_text(LINK)
            run(
                [args.clang, *flags, "-c", output / "main.c", "-o", output / "main.o"],
                output,
                "driver-compile",
            )
            run(
                [
                    args.clang,
                    *flags,
                    "-c",
                    output / "start.S",
                    "-o",
                    output / "start.o",
                ],
                output,
                "startup-compile",
            )
            elf = output / "proof.elf"
            run(
                [
                    args.linker,
                    "-T",
                    output / "link.ld",
                    output / "start.o",
                    output / "main.o",
                    *objects,
                    "-o",
                    elf,
                ],
                output,
                "link",
            )
            isa = f"rv64gcv_zvl{vlen}b"
            run(
                [
                    args.spike,
                    *args.spike_option,
                    "--isa=" + isa,
                    "-p1",
                    "-m0x80000000:0x8000000",
                    elf,
                ],
                output,
                "native",
                timeout=60,
            )
            report["cases"].append(
                {
                    "vlen_bits": vlen,
                    "pass_host": True,
                    "pass_native": True,
                    "native_isa": isa,
                    "ELF_sha256": sha(elf),
                    "driver_sha256": sha(output / "main.c"),
                    "startup_sha256": sha(output / "start.S"),
                    "linker_script_sha256": sha(output / "link.ld"),
                    "host_library_sha256": sha(library),
                    "fixture_count": 2,
                    "patterns_per_fixture": 8,
                    "full_output_words_checked": 2 * 8 * 195,
                    "full_storage_words_checked": 8 * (195 + 207),
                    "actual_vlenb_assertion": vlen // 8,
                    "post_vector_leaf_VL_assertion": "equal declared feature NR; no claim about every intermediate VL",
                    "raw_vtype_trace": None,
                    "fixtures": fixtures,
                }
            )
        if any(sha(Path(path)) != digest for path, digest in tools.items()):
            raise RuntimeError("tool bytes changed during execution")
        report["pass"] = True
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        report["error"] = repr(error)
    report["source_sha256"] = {
        str(path): sha(path)
        for path in (
            Path(__file__),
            ROOT / "src/rvv/packing/borrowed_i8_leaf.py",
            ROOT / "src/rvv/backend.py",
            ROOT / "tests/prove_im2col_stream.py",
        )
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report.get(key) for key in ("pass", "error")}))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
