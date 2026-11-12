"""Explicit prepared upstream-MLIR to LLVM-IR service; no image construction.

This narrow adapter selects the actual OOT pipeline and shipped schedule. It
does not capture a model, prepare quantization, claim a platform ABI, or link an
executable. The existing Merlin native driver executes and records the passes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

name = "rvv"
_OPTIONS = {"rvv.data-layout", "rvv.index-bits", "rvv.schedule", "rvv.timeout"}
_SERVICE_IMPORT_SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def codegen_provider():
    """The installed service's explicitly selected vector policy implementation."""
    if (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        != _SERVICE_IMPORT_SOURCE
    ):
        raise ValueError(
            "selected vector service changed after import; use a fresh provider process"
        )
    from .src.rvv import backend

    return backend


def validate(request) -> None:
    from merlin.compiler.lower.model import toolchain

    from .src.rvv import backend

    if request.source.kind != "upstream-mlir":
        raise ValueError(
            "RVV service requires explicitly preprocessed upstream-mlir input"
        )
    options = request.options
    if any(
        value is not None for value in (options.system, options.platform, options.harts)
    ):
        raise ValueError(
            "RVV LLVM-IR emission does not build a platform image or schedule harts"
        )
    values = options.parameters
    if set(values) - _OPTIONS:
        raise ValueError(
            f"unsupported RVV parameters: {sorted(set(values) - _OPTIONS)}"
        )
    layout = values.get("rvv.data-layout")
    if not isinstance(layout, str) or not layout.strip():
        raise ValueError("RVV emission requires an explicit rvv.data-layout")
    if type(values.get("rvv.index-bits")) is not int or values[
        "rvv.index-bits"
    ] not in (32, 64):
        raise ValueError("rvv.index-bits must explicitly select 32 or 64")
    schedule = values.get("rvv.schedule", backend.DEFAULT_SCHEDULE)
    root = Path(__file__).parent / "schedules"
    if not isinstance(schedule, str) or schedule not in {
        p.name for p in root.iterdir() if p.is_dir()
    }:
        raise ValueError("rvv.schedule must name a shipped provider schedule")
    backend.schedule_path(schedule)
    timeout = values.get("rvv.timeout", 7200)
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("rvv.timeout must be a positive integer")
    driver = toolchain._env("MERLIN_COMPILER_DRIVER", "python")
    if driver == "stock-cli":
        for tool in (
            toolchain.llvm_install() / "bin" / "mlir-opt",
            toolchain.mlir_translate(),
        ):
            if not tool.is_file():
                raise FileNotFoundError(
                    f"selected stock-cli compiler tool is unavailable: {tool}"
                )
    elif driver == "python":
        interpreter = toolchain.compiler_python()
        if not interpreter.is_file():
            raise FileNotFoundError(
                f"native torch-mlir compiler interpreter is unavailable: {interpreter}"
            )
    else:
        raise ValueError(f"unknown upstream compiler driver: {driver!r}")


def compile(request):
    from merlin.compiler.artifact import Artifact, Compilation
    from merlin.compiler.lower.llvm.passes import finalize_pipeline
    from merlin.compiler.lower.native.pipeline import lower_to_llvm_ir

    from .src.rvv import backend

    validate(request)
    output = request.output
    if output.exists():
        raise FileExistsError(
            f"RVV compilation requires a fresh output directory: {output}"
        )
    output.mkdir(parents=True)
    values = request.options.parameters
    selected = values.get("rvv.schedule", backend.DEFAULT_SCHEDULE)
    schedule = output / "schedule.mlir"
    schedule.write_bytes(
        (backend.schedule_path(selected) / "schedule.mlir").read_bytes()
    )
    pipeline = backend.build_pipeline(schedule)
    finalize_pipeline(pipeline)  # complete terminal conversion remains mandatory
    pipeline_path = output / "pipeline.txt"
    pipeline_path.write_text(pipeline + "\n", encoding="utf-8")
    try:
        text = lower_to_llvm_ir(
            request.source.text,
            workdir=output / "native",
            pipeline=pipeline,
            data_layout=values["rvv.data-layout"],
            index_bits=values["rvv.index-bits"],
            timeout=values.get("rvv.timeout", 7200),
            features=frozenset(),
        )
        llvm = output / "module.ll"
        llvm.write_text(text, encoding="utf-8")
        products = [
            Artifact.from_file(llvm, kind="llvm-ir", role="region"),
            Artifact.from_file(schedule, kind="transform-schedule", role="schedule"),
            Artifact.from_file(pipeline_path, kind="pass-pipeline", role="pipeline"),
        ]
        for path in sorted((output / "native").rglob("*")):
            if path.is_file():
                products.append(
                    Artifact.from_file(path, kind="native-artifact", role="trace")
                )
        return Compilation(
            output,
            name,
            request.source.sha256,
            tuple(products),
            metadata={
                "scope": "prepared_upstream_mlir_to_llvm_ir",
                "schedule": selected,
                "data_layout": values["rvv.data-layout"],
                "index_bits": values["rvv.index-bits"],
                "execution": "not_run",
                "qualification": "not_established",
            },
        )
    except Exception as exc:
        (output / "failure.json").write_text(
            json.dumps(
                {"status": "failed", "type": type(exc).__name__, "reason": str(exc)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
