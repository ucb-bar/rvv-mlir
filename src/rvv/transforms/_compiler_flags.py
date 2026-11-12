"""Shared RV64 architecture validation and compiler flag selection."""

import re

MARCH = re.compile(r"rv64[a-z0-9_]+")


def with_march(flags: tuple[str, ...], march: str) -> tuple[str, ...]:
    """Validate the RV64 ISA and substitute its compiler flag."""
    if not isinstance(march, str) or not MARCH.fullmatch(march):
        raise ValueError("march must be an RV64 ISA string")
    return tuple(f"-march={march}" if flag.startswith("-march=") else flag for flag in flags)
