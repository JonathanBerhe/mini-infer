"""Static check of Hopper SASS: writes to registers that an in-flight `wgmma` reads.

A `wgmma` (SASS `HGMMA`) with its left operand in registers reads those
registers, and its accumulator, while it runs. Code that overwrites them
before the `WARPGROUP.DEPBAR` that retires the `wgmma` gives wrong results
that change from run to run. ptxas does not warn about it. This check finds
each loop around an `HGMMA`, runs its body twice (so the loop's back edge
counts), and reports every other instruction that writes a protected
register.
"""

from __future__ import annotations

import re

_LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
_BRANCH = re.compile(r"^(?:@!?U?P\w+ )?BRA (?:!?U?P\w+, )?(0x[0-9a-f]+)")
# Opcodes whose first operand is not a register they write.
_NO_DEST = (
    "ST",
    "RED",
    "BAR",
    "BRA",
    "BSYNC",
    "BSSY",
    "EXIT",
    "SYNCS",
    "UTMA",
    "WARPSYNC",
    "NOP",
    "FENCE",
    "MEMBAR",
    "CCTL",
    "ERRBAR",
    "DEPBAR",
    "WARPGROUP",
    "CALL",
    "RET",
    "YIELD",
    "USETMAXREG",
    "HGMMA",
)


def _strip_guard(text: str) -> str:
    return text.split(None, 1)[1] if text.startswith("@") else text


def _span(first: int, count: int) -> set[int]:
    return set(range(first, first + count))


def _written(text: str) -> set[int]:
    """General registers an instruction writes (its first operand)."""
    opcode, _, operands = _strip_guard(text).partition(" ")
    if opcode.startswith(_NO_DEST):
        return set()
    match = re.fullmatch(r"R(\d+)(?:\.\w+)*", operands.split(",")[0].strip())
    if match is None:
        return set()
    width = 4 if ".128" in opcode else 2 if ".64" in opcode else 1
    return _span(int(match.group(1)), width)


def _hgmma_reads(text: str) -> set[int]:
    """Accumulator and register-operand registers of one `HGMMA`."""
    instruction = _strip_guard(text)
    opcode, _, operands = instruction.partition(" ")
    shape = re.match(r"HGMMA\.64x(\d+)x", opcode)
    assert shape is not None, instruction
    parts = [part.strip() for part in operands.split(",")]
    # An fp32 64xN accumulator takes N / 2 registers per thread, a 64x16
    # 16-bit left operand 4.
    reads = _span(int(parts[0][1:]), int(shape.group(1)) // 2)
    if parts[1].startswith("R"):
        reads |= _span(int(parts[1][1:].split(".")[0]), 4)
    return reads


def wgmma_operand_hazards(sass: str) -> list[str]:
    """Instructions in an `HGMMA` loop that write a register an in-flight `HGMMA` reads."""
    code = [(int(m.group(1), 16), m.group(2)) for m in map(_LINE.search, sass.splitlines()) if m]
    loops: set[tuple[int, int]] = set()
    for index, (address, text) in enumerate(code):
        if not _strip_guard(text).startswith("HGMMA"):
            continue
        innermost: tuple[int, int] | None = None
        for later_address, later in code[index:]:
            branch = _BRANCH.match(later)
            if branch is None:
                continue
            target = int(branch.group(1), 16)
            if target <= address and (
                innermost is None or later_address - target < innermost[1] - innermost[0]
            ):
                innermost = (target, later_address)
        if innermost is not None:
            loops.add(innermost)

    hazards: list[str] = []
    for start, end in sorted(loops):
        body = [(address, text) for address, text in code if start <= address <= end]
        in_flight: list[set[int]] = []
        open_group: set[int] = set()
        for _iteration in range(2):
            for address, text in body:
                instruction = _strip_guard(text)
                if instruction.startswith("HGMMA"):
                    open_group |= _hgmma_reads(text)
                    if "gsb0" in instruction:
                        in_flight.append(open_group)
                        open_group = set()
                elif instruction.startswith("WARPGROUP.DEPBAR"):
                    keep = int(instruction.rsplit(",", 1)[1].strip(), 16)
                    in_flight = in_flight[len(in_flight) - keep :] if keep else []
                else:
                    clash = _written(text) & set().union(open_group, *in_flight)
                    hazard = f"{address:#06x} {text} writes R{min(clash, default=0)}"
                    if clash and hazard not in hazards:
                        hazards.append(hazard)
    return hazards
