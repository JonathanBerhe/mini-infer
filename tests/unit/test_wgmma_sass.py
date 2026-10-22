"""The SASS check for registers of in-flight `wgmma`s (`_wgmma_sass.py`), on short loops."""

from __future__ import annotations

from tests.unit._wgmma_sass import wgmma_operand_hazards


def _sass(*lines: tuple[int, str]) -> str:
    return "\n".join(
        f"        /*{address:04x}*/                   {text} ;" for address, text in lines
    )


def test_reports_an_operand_rewritten_while_its_wgmma_runs() -> None:
    # One register operand for every K step: the next step's load overwrites
    # it while the previous step's wgmma still runs (the wait keeps 1).
    sass = _sass(
        (0x100, "LDS.U16 R88, [R2]"),
        (0x110, "WARPGROUP.ARRIVE"),
        (0x120, "HGMMA.64x128x16.F32.BF16 R24, R88, gdesc[UR4], R24, gsb0"),
        (0x130, "WARPGROUP.DEPBAR.LE gsb0, 0x1"),
        (0x140, "@P0 BRA 0x100"),
        (0x150, "WARPGROUP.DEPBAR.LE gsb0, 0x0"),
    )
    hazards = wgmma_operand_hazards(sass)
    assert hazards == ["0x0100 LDS.U16 R88, [R2] writes R88"]


def test_accepts_alternating_operands() -> None:
    sass = _sass(
        (0x100, "LDS.U16 R88, [R2]"),
        (0x110, "WARPGROUP.ARRIVE"),
        (0x120, "HGMMA.64x128x16.F32.BF16 R24, R88, gdesc[UR4], R24, gsb0"),
        (0x130, "WARPGROUP.DEPBAR.LE gsb0, 0x1"),
        (0x140, "LDS.U16 R92, [R2]"),
        (0x150, "WARPGROUP.ARRIVE"),
        (0x160, "HGMMA.64x128x16.F32.BF16 R24, R92, gdesc[UR4], R24, gsb0"),
        (0x170, "WARPGROUP.DEPBAR.LE gsb0, 0x1"),
        (0x180, "@P0 BRA 0x100"),
    )
    assert wgmma_operand_hazards(sass) == []


def test_reports_an_accumulator_write_before_the_wait() -> None:
    sass = _sass(
        (0x100, "WARPGROUP.ARRIVE"),
        (0x110, "HGMMA.64x64x16.F32.BF16 R24, gdesc[UR8], R24, gsb0"),
        (0x120, "FADD R30, R30, 1"),
        (0x130, "WARPGROUP.DEPBAR.LE gsb0, 0x0"),
        (0x140, "FADD R30, R30, 1"),
        (0x150, "@!P1 BRA 0x100"),
    )
    assert wgmma_operand_hazards(sass) == ["0x0120 FADD R30, R30, 1 writes R30"]
