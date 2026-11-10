"""Prove generated exact-BF16 linear output against Merlin scalar output in Spike."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHAPES = ((4, 2048, 256), (1, 2048, 1000), (8, 512, 513),
          (5, 129, 77), (3, 37, 45), (2, 1, 17), (4, 8, 16), (1, 15, 33))
VLENS = (128, 256, 512)


def scalar_source(source: str, m: int, k: int, n: int) -> str:
    """Apply the emitter's literal template substitutions without importing Merlin."""
    template = re.search(r'implementation = r"""(.*?)"""\n', source, re.S)
    replacements = re.search(r'replacements = \{(.*?)\}\n', source, re.S)
    if template is None or replacements is None:
        raise ValueError("unsupported Merlin scalar emitter template")
    sizes = (2*m*k, 2*k*n, 4*m*n)
    values = dict(PREFIX="merlin_ref", MAXSIZE=str(max(sizes)), ASIZE=str(sizes[0]),
                  BSIZE=str(sizes[1]), SEEDSIZE=str(sizes[2]), CSIZE=str(2*m*n),
                  MDIM=str(m), KDIM=str(k), NDIM=str(n))
    if re.findall(r'"([A-Z]+)":', replacements.group(1)) != list(values):
        raise ValueError("unsupported Merlin scalar template replacements")
    result = template.group(1)
    for key, value in values.items():
        result = result.replace(key, value)
    return result


def run(command: list[str], log: Path) -> str:
    result = subprocess.run(list(map(str, command)),
                            capture_output=True, text=True, timeout=600)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compiler", "objdump", "spike", "pk", "merlin-emitter", "output",
                 "mlir-opt", "mlir-translate"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    codegen = parser.add_mutually_exclusive_group()
    codegen.add_argument("--clang", type=Path,
                         help="LLVM code generator (default: RVV_EXACT_CLANG or clang on PATH)")
    codegen.add_argument("--llc", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--march", default="rv64gcv")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    module_path = Path(__file__).resolve().parents[1] / "src/rvv/transforms/exact_bf16_linear.py"
    sys.path.insert(0, str(module_path.parents[2]))
    from rvv.transforms import exact_bf16_linear as api
    emitter = args.merlin_emitter.read_text()
    flags = [*api.compiler_flags(args.march), "-std=c11"]
    clang = args.clang or (None if args.llc else Path(os.environ.get("RVV_EXACT_CLANG", "clang")))
    commands = []
    audit = []
    for m, k, n, vlen in ((m, k, n, vlen) for m, k, n in SHAPES for vlen in VLENS):
        work = args.output / f"{m}x{k}x{n}" / f"v{vlen}"
        work.mkdir(parents=True)
        scalar = work / "scalar.c"
        target = api.PanelTarget(
            vlen_bits=vlen, elen_bits=64, operand_bits=16,
            accumulator_bits=32, operand_lmul=1,
            workspace_bytes=max(4*k*n, 1), max_mr=4,
        )
        target_march = re.sub(r"_zvl\d+b(?=_|$)", "", args.march) + f"_zvl{vlen}b"
        report = api.compile_kernel(
            api.LinearSpec(m=m, k=k, n=n), work=work / "production",
            compiler=args.compiler, mlir_opt=args.mlir_opt,
            mlir_translate=args.mlir_translate, clang=clang, llc=args.llc,
            enabled=True, march=target_march, target=target,
        )
        generated = Path(report["source"])
        scalar.write_text(scalar_source(emitter, m, k, n))
        obj, elf = Path(report["object"]), work / "equality.elf"
        dis = run([args.objdump, "-d", obj], work / "generated.dis")
        # Include destructive and widening vector FMA spellings and scalar FMA.
        fused = re.findall(r'\b(?:vf(?:n?m(?:acc|add|sac|sub)|w[n]?m(?:acc|add|sac|sub))|f[n]?m(?:add|sub))\.[a-z]+', dis)
        if fused:
            raise AssertionError(f"fused arithmetic in {obj}: {fused}")
        segment = dis.count("vlsseg8e16.v")
        strided = dis.count("vlse16.v")
        if k > 1 and not (strided or segment):
            raise AssertionError(f"missing strided original-weight loads in {obj}")
        if not re.search(r"vfmul\.v[fv]", dis) or not re.search(r"vfadd\.v[fv]", dis):
            raise AssertionError(f"missing separate multiply/add in {obj}")
        audit.append(dict(shape=[m,k,n], vlen=vlen, lanes=report["lanes"],
                          rows=report["rows"],
                          fused_ops=len(fused), segment_loads=segment, strided_loads=strided, load_strategy=report["load_strategy"],
                          commands=report["commands"],
                          source_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
                          object_sha256=hashlib.sha256(obj.read_bytes()).hexdigest()))
        run([args.compiler, *flags, f"-DM_={m}", f"-DK_={k}", f"-DN_={n}",
             Path(__file__).with_name("exact_bf16_linear_harness.c"), scalar, obj, "-o", elf],
            work / "link.log")
        for mode in range(15):
            seeds = (1,) if k >= 512 or mode >= 8 else (1, 2)
            for seed in seeds:
                commands.append((work, elf, vlen, mode, seed))

    def execute(job):
        work, elf, vlen, mode, seed = job
        log = work / f"v{vlen}-mode{mode}-seed{seed}.log"
        stdout = run([args.spike, f"--isa=rv64gcv_zicntr_zvl{vlen}b", args.pk,
                      elf, str(mode), str(seed), str(vlen)], log)
        lines = [line for line in stdout.splitlines() if line.startswith("PASS ")]
        if len(lines) != 1:
            raise AssertionError(f"missing passing result in {log}")
        return lines[0]

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(execute, commands))
    (args.output / "results.txt").write_text("\n".join(results) + "\n")
    summary = dict(runs=len(results), pass_count=len(results),
                   scalar_emitter_sha256=hashlib.sha256(emitter.encode()).hexdigest(),
                   objects=audit, retired_instructions=[])
    for line in results:
        fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
        if fields["mode"] == "0" and fields["seed"] == "1":
            summary["retired_instructions"].append(dict(
                shape=[int(fields[x]) for x in ("M","K","N")], vlen=int(fields["vlen"]),
                scalar=int(fields["instr_merlin"]), generated=int(fields["instr_generated"])))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(runs=len(results), pass_count=len(results), fused_ops=0)))


if __name__ == "__main__":
    main()
