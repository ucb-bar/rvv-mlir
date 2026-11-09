import pytest
from borrowed_i8_proof_audit import audit


@pytest.mark.parametrize(
    "symbol",
    [
        "malloc",
        "calloc",
        "realloc",
        "aligned_alloc",
        "posix_memalign",
        "free",
        "memrefCopy",
        "memcpy",
        "memmove",
        "memset",
        "bcopy",
        "bzero",
        "llvm.memcpy.p0.p0.i64",
        "llvm.memcpy.inline.p0.p0.i64",
        "llvm.memcpy.element.unordered.atomic.p0.p0.i64",
        "llvm.memmove.p0.p0.i32",
        "llvm.memset.p0.i64",
    ],
)
def test_allocation_copy_and_zeroing_symbols_refuse(symbol):
    result = audit(
        f'define void @leaf() {{\n call void @"{symbol}"(ptr null)\n ret void\n}}'
    )
    assert not result["pass"]
    assert any(v.get("symbol") == symbol for v in result["violations"])


def test_dead_declaration_refuses_without_claiming_it_executed():
    result = audit("declare void @free(ptr)\ndefine void @leaf(){ret void}")
    assert not result["pass"]
    assert result["scope"].startswith("conservative")


@pytest.mark.parametrize(
    "instruction",
    [
        "call void %callback(ptr null)",
        "invoke void @opaque() to label %ok unwind label %bad",
        'call void asm sideeffect "", ""()',
        "call void @opaque(\n ptr null)",
        "%x = alloca i32, align 4",
    ],
)
def test_unqualified_effects_refuse(instruction):
    assert not audit("define void @leaf(){\n" + instruction + "\nret void\n}")["pass"]


def test_closed_direct_calls_and_arithmetic_pass():
    text = """define void @leaf(){
      %x = add i32 1, 2
      ret void
    }
    define void @_mlir_ciface_leaf(){
      call void @leaf()
      ret void
    }"""
    assert audit(text)["pass"]


def test_comments_are_not_memory_effects():
    assert audit(
        "; call void @free(ptr null)\ndefine void @leaf(){ ret void ; alloca\n}"
    )["pass"]


def test_multiline_direct_internal_target_is_supported():
    assert audit(
        "define void @leaf(){ret void}\ndefine void @wrapper(){\ncall void\n  @leaf()\nret void\n}"
    )["pass"]


def test_adjacent_unknown_call_cannot_hide_after_an_internal_call():
    result = audit(
        "define void @leaf(){ret void}\ndefine void @wrapper(){\ncall void @leaf()\n call void @opaque()\nret void\n}"
    )
    assert not result["pass"]
    assert any(v.get("symbol") == "opaque" for v in result["violations"])


def test_inline_assembly_string_cannot_fabricate_an_internal_call():
    result = audit(
        'define void @leaf(){ret void}\ndefine void @wrapper(){\ncall void asm sideeffect "call @leaf()", ""()\nret void\n}'
    )
    assert not result["pass"]
    assert any(v["kind"] == "unsupported_assembly" for v in result["violations"])


def test_module_assembly_also_refuses():
    assert not audit('module asm "addi x1,x1,1"\ndefine void @leaf(){ret void}')["pass"]


def test_string_cannot_fabricate_a_function_definition():
    result = audit(
        'source_filename = "define void @opaque()"\ndefine void @wrapper(){\ncall void @opaque()\nret void\n}'
    )
    assert not result["pass"]


@pytest.mark.parametrize("symbol", ['@"fr\\65e"', '%"local"', '@"leaf"'])
def test_quoted_or_escaped_symbols_refuse_as_unsupported_grammar(symbol):
    result = audit("declare void " + symbol + "()")
    assert not result["pass"]
    assert any(v["kind"] == "unsupported_quoted_symbol" for v in result["violations"])


def test_irrelevant_strings_do_not_create_memory_or_assembly_effects():
    assert audit(
        'source_filename = "asm ; call @free()"\ndefine void @leaf(){ret void}\n!0 = !{!"define void @evil()"}'
    )["pass"]


def test_asm_identifier_is_not_inline_assembly():
    assert audit(
        "define void @asm(){ret void}\ndefine void @wrapper(){\ncall void @asm()\nret void\n}"
    )["pass"]


def test_local_define_identifier_cannot_fabricate_a_definition():
    assert not audit(
        "define void @wrapper(){\n %define = call i32 @opaque()\nret void\n}"
    )["pass"]
