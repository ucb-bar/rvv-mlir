"""Conservative LLVM closure gate for the compute-only borrowed-leaf proof.

This admits the plain-symbol grammar produced by these compiler leaf proofs,
not arbitrary LLVM. Inline assembly, quoted symbols, unknown and indirect calls
refuse. Even dead declarations of forbidden helpers refuse this strict corpus.
"""

from __future__ import annotations

import re

_NAME = r"[-a-zA-Z$._0-9]+"
_SYMBOL = re.compile(r"@(" + _NAME + r")")
_DIRECT_TARGET = re.compile(r"@(" + _NAME + r")\s*\(")
_INDIRECT_TARGET = re.compile(r"%(" + _NAME + r")\s*\(")
_DEFINITION = re.compile(r"^\s*define\b[^@]*@(" + _NAME + r")\s*\(", re.MULTILINE)
_MEMORY_HELPERS = frozenset(
    {
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
    }
)


def _lexical_code(text: str) -> tuple[str, list[dict]]:
    """Mask comments/strings without letting their text invent IR tokens.

    Quoted LLVM symbols are legal LLVM but outside this conservative proof
    grammar. Refuse them rather than interpreting LLVM's escaped-name syntax.
    Newlines remain in place for diagnostic locations.
    """
    code = list(text)
    violations = []
    index = 0
    while index < len(text):
        if text[index] == ";":
            while index < len(text) and text[index] != "\n":
                code[index] = " "
                index += 1
            continue
        if text[index] != '"':
            index += 1
            continue
        start = index
        quoted_symbol = index > 0 and text[index - 1] in "@%"
        index += 1
        while index < len(text):
            if text[index] == "\\":
                index += 2
            elif text[index] == '"':
                index += 1
                break
            else:
                index += 1
        else:
            violations.append(
                {"kind": "unterminated_string", "line": text.count("\n", 0, start) + 1}
            )
        if quoted_symbol:
            violations.append(
                {
                    "kind": "unsupported_quoted_symbol",
                    "line": text.count("\n", 0, start) + 1,
                    "symbol": text[start + 1 : index - 1],
                }
            )
        for position in range(start, min(index, len(text))):
            if text[position] != "\n":
                code[position] = " "
    return "".join(code), violations


def audit(text: str) -> dict:
    """Refuse allocation, copies, zeroing, stack allocation and opaque calls.

    Internal direct calls are allowed only when their definitions occur in this
    same module; all bodies are checked. Memset is refused because this leaf
    contract reads caller-seeded C and admits no initialization helper.
    """
    if not isinstance(text, str):
        raise TypeError("LLVM text must be a string")
    code, violations = _lexical_code(text)
    for line_number, line in enumerate(code.splitlines(), 1):
        if re.search(r"\balloca\b", line):
            violations.append({"kind": "stack_allocation", "line": line_number})
        if re.search(r"(?<![-@%A-Za-z0-9$._])asm(?![-A-Za-z0-9$._])", line):
            violations.append({"kind": "unsupported_assembly", "line": line_number})
        for match in _SYMBOL.finditer(line):
            name = match.group(1)
            if name in _MEMORY_HELPERS or name.startswith(
                ("llvm.memcpy.", "llvm.memmove.", "llvm.memset.")
            ):
                violations.append(
                    {
                        "kind": "forbidden_memory_symbol",
                        "line": line_number,
                        "symbol": name,
                    }
                )

    definitions = {m.group(1) for m in _DEFINITION.finditer(code)}
    # A direct target may be on the next line. Never consume another call while
    # extending that spelling: every call must receive its own admission check.
    lines = code.splitlines()
    for index, line in enumerate(lines):
        for match in re.finditer(r"\b(?:call|invoke|callbr)\b", line):
            candidate = line[match.end() :]
            direct = _DIRECT_TARGET.search(candidate)
            indirect = _INDIRECT_TARGET.search(candidate)
            if direct is None and indirect is None and index + 1 < len(lines):
                continuation = lines[index + 1]
                if not re.search(r"\b(?:call|invoke|callbr)\b", continuation):
                    candidate += "\n" + continuation
                    direct = _DIRECT_TARGET.search(candidate)
                    indirect = _INDIRECT_TARGET.search(candidate)
            is_direct = direct is not None and (
                indirect is None or direct.start() < indirect.start()
            )
            name = direct.group(1) if is_direct else None
            if name not in definitions:
                violations.append(
                    {"kind": "unqualified_call", "line": index + 1, "symbol": name}
                )
    return {
        "pass": not violations,
        "violations": violations,
        "scope": "conservative plain-symbol returned-LLVM closure admission; not arbitrary LLVM parsing or execution reachability",
    }
