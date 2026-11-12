"""Service admission/delegation and optional stock LLVM compilation; no execution claim."""

import json
import subprocess
from pathlib import Path

import pytest
from merlin import api
from merlin.compiler.request import CompileRequest
from merlin.development.targetgen.plugins import load_module

ROOT = Path(__file__).resolve().parents[1]


def test_service_rejects_unimplemented_input_options_and_missing_native_toolchain(
    tmp_path, monkeypatch
):
    from merlin.compiler.lower.model import toolchain

    monkeypatch.setenv("MERLIN_COMPILER_DRIVER", "python")
    provider = load_module(ROOT, "service.py", package_name="rvv_service_contract")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="preprocessed"):
        api.compile("module {}", backend=provider, output=output)
    source = api.Source("module {}", kind="upstream-mlir")
    with pytest.raises(ValueError, match="data-layout"):
        api.compile(source, backend=provider, output=output)
    with pytest.raises(ValueError, match="platform image"):
        api.compile(
            source,
            backend=provider,
            output=output,
            options=api.CompileOptions(platform="zephyr"),
        )
    with pytest.raises(ValueError, match="unsupported"):
        api.compile(
            source,
            backend=provider,
            output=output,
            options=api.CompileOptions(parameters={"rvv.unknown": True}),
        )
    monkeypatch.setattr(
        toolchain, "compiler_python", lambda: tmp_path / "missing-python"
    )
    with pytest.raises(FileNotFoundError, match="native torch-mlir"):
        api.compile(
            source,
            backend=provider,
            output=output,
            options=api.CompileOptions(
                parameters={"rvv.data-layout": "e-p:64:64", "rvv.index-bits": 64}
            ),
        )
    assert not output.exists()


def test_service_delegates_exact_oot_pipeline_and_selected_layout(
    tmp_path, monkeypatch
):
    from merlin.compiler.lower.native import pipeline
    from merlin.compiler.lower.model import toolchain

    monkeypatch.setenv("MERLIN_COMPILER_DRIVER", "python")
    provider = load_module(ROOT, "service.py", package_name="rvv_service_delegation")
    interpreter = tmp_path / "selected-python"
    interpreter.write_text("test-only native call is intercepted below")
    monkeypatch.setattr(toolchain, "compiler_python", lambda: interpreter)
    calls = []

    def lower(source, **kwargs):
        calls.append((source, kwargs))
        kwargs["workdir"].mkdir()
        (kwargs["workdir"] / "lowering_recipe.json").write_text(
            '{"test": "delegation-only"}'
        )
        return "; test-only delegation result\n"

    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", lower)
    source = api.Source("builtin.module {}", kind="upstream-mlir")
    options = api.CompileOptions(
        parameters={"rvv.data-layout": "e-p:64:64", "rvv.index-bits": 64}
    )
    request = CompileRequest(source, tmp_path / "output", options)
    result = api.compile(
        source, backend=provider, output=request.output, options=options
    )
    assert result.backend == "rvv" and result.deployment is None
    assert result.source_sha256 == source.sha256
    assert calls[0][0] == source.text
    assert calls[0][1]["data_layout"] == "e-p:64:64"
    assert calls[0][1]["index_bits"] == 64
    assert "transform-preload-library" in calls[0][1]["pipeline"]
    assert "convert-vector-to-llvm" in calls[0][1]["pipeline"]
    assert {item.kind for item in result.artifacts} == {
        "llvm-ir",
        "transform-schedule",
        "pass-pipeline",
        "native-artifact",
    }
    assert api.inspect(result)["metadata"]["execution"] == "not_run"


@pytest.mark.parametrize(
    "selection", ["stock-cli", "missing-opt", "missing-translate", "unknown"]
)
def test_service_admits_only_the_selected_stock_transport_tools(
    tmp_path, monkeypatch, selection
):
    from merlin.compiler.lower.model import toolchain

    provider = load_module(
        ROOT, "service.py", package_name="rvv_selected_stock_transport"
    )
    prefix = tmp_path / "llvm"
    (prefix / "bin").mkdir(parents=True)
    for tool in ("mlir-opt", "mlir-translate"):
        if selection != "missing-" + ("opt" if tool == "mlir-opt" else "translate"):
            (prefix / "bin" / tool).write_text("tool discovery only; never executed")
    monkeypatch.setattr(toolchain, "llvm_install", lambda: prefix)
    monkeypatch.setattr(
        toolchain, "mlir_translate", lambda: prefix / "bin/mlir-translate"
    )

    def no_python():
        raise AssertionError("stock-cli must not select the Python transport")

    monkeypatch.setattr(toolchain, "compiler_python", no_python)
    monkeypatch.setenv(
        "MERLIN_COMPILER_DRIVER", "other" if selection == "unknown" else "stock-cli"
    )
    request = CompileRequest(
        api.Source("builtin.module {}", kind="upstream-mlir"),
        tmp_path / "output",
        api.CompileOptions(
            parameters={"rvv.data-layout": "e-p:64:64", "rvv.index-bits": 64}
        ),
    )
    if selection == "stock-cli":
        provider.validate(request)
    else:
        with pytest.raises(
            (ValueError, FileNotFoundError), match="unknown upstream|selected stock-cli"
        ):
            provider.compile(request)
    assert not request.output.exists()


@pytest.mark.parametrize("program", ["scalar", "matmul"])
def test_selected_stock_pipeline_compiles_and_verifies_llvm_without_python_transport(
    tmp_path, monkeypatch, program
):
    from merlin.compiler.lower.model import toolchain

    prefix = toolchain.llvm_install()
    opt = prefix / "bin/opt"
    if not all(
        path.is_file()
        for path in (prefix / "bin/mlir-opt", toolchain.mlir_translate(), opt)
    ):
        pytest.skip("explicit stock LLVM/MLIR install and verifier are unavailable")
    monkeypatch.setenv("MERLIN_COMPILER_DRIVER", "stock-cli")
    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", str(tmp_path / "unselected-python"))
    # This transport does not implement the Python fusion callback. Select its
    # existing explicit unguarded mode; numerical qualification remains separate.
    monkeypatch.setenv("MERLIN_FUSION_GUARD", "0")
    provider = load_module(ROOT, "service.py", package_name="rvv_stock_native_service")
    source = (
        "builtin.module { func.func @forward(%a: f32, %b: f32) -> f32 { "
        "%r = arith.addf %a, %b : f32 func.return %r : f32 } }"
        if program == "scalar"
        else """builtin.module {
          func.func @forward(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
            %z = arith.constant 0.0 : f32
            %e = tensor.empty() : tensor<2x2xf32>
            %f = linalg.fill ins(%z : f32) outs(%e : tensor<2x2xf32>) -> tensor<2x2xf32>
            %r = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>)
                 outs(%f : tensor<2x2xf32>) -> tensor<2x2xf32>
            func.return %r : tensor<2x2xf32>
          }
        }"""
    )
    layout = "e-p:64:64-i64:64-n32:64"
    options = api.CompileOptions(
        parameters={
            "rvv.data-layout": layout,
            "rvv.index-bits": 64,
            "rvv.timeout": 120,
        }
    )
    output = tmp_path / program
    result = api.compile(
        api.Source(source, kind="upstream-mlir"),
        backend=provider,
        output=output,
        options=options,
    )
    verified = subprocess.run(
        [str(opt), "-passes=verify", "-disable-output", str(output / "module.ll")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    assert f'target datalayout = "{layout}"' in (output / "module.ll").read_text()
    record = json.loads((output / "native/lowering_recipe.json").read_text())
    assert record["status"] == "returned"
    assert [Path(row["argv"][0]).name for row in record["commands"]] == [
        "mlir-opt",
        "mlir-translate",
    ]
    assert "index-bitwidth=64" in record["commands"][0]["argv"][2]
    assert result.deployment is None
    assert result.metadata["execution"] == "not_run"
    assert result.metadata["qualification"] == "not_established"
    if program == "scalar":
        # An unsupported selected pipeline fails through the actual stock tool;
        # the absent Python interpreter must never become another attempt.
        from importlib import import_module

        backend = import_module(provider.__package__ + ".src.rvv.backend")
        original = backend.build_pipeline
        monkeypatch.setattr(
            backend,
            "build_pipeline",
            lambda *a, **k: "unregistered-rvv-test-pass," + original(*a, **k),
        )
        failed = tmp_path / "unsupported"
        with pytest.raises(RuntimeError, match="stock-cli"):
            api.compile(
                api.Source(source, kind="upstream-mlir"),
                backend=provider,
                output=failed,
                options=options,
            )
        assert (
            "unregistered-rvv-test-pass"
            in json.loads((failed / "failure.json").read_text())["reason"]
        )
        assert not (failed / "compilation.json").exists()
