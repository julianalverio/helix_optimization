from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gemmi

from .constants import DSSP_CHAR_TO_SS8, SS8_NULL, SS8_TO_SS3

logger = logging.getLogger(__name__)

SsKey = tuple[str, int]
SsCodes = tuple[int, int]


@dataclass
class DsspOutcome:
    ok: bool
    ss_map: dict[SsKey, SsCodes]
    reason: str


def _ss_char_to_codes(ch: str) -> SsCodes:
    key = " " if ch in (".", "?", "", " ") else ch
    ss8 = DSSP_CHAR_TO_SS8.get(key, SS8_NULL)
    return SS8_TO_SS3[ss8], ss8


def run_dssp(structure: gemmi.Structure, executable: str = "mkdssp") -> DsspOutcome:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_path = tmp / "in.cif"
        out_path = tmp / "out.cif"
        structure.make_mmcif_document().write_file(str(in_path))
        try:
            result = subprocess.run(
                [executable, "--output-format", "mmcif", str(in_path), str(out_path)],
                capture_output=True, text=True, timeout=180,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return DsspOutcome(False, {}, f"subprocess_failed: {type(exc).__name__}")
        if result.returncode != 0:
            return DsspOutcome(False, {}, f"exit_{result.returncode}: {result.stderr[:200].strip()}")
        if not out_path.exists():
            return DsspOutcome(False, {}, "no_output_file")
        try:
            doc = gemmi.cif.read(str(out_path))
            block = doc.sole_block()
            table = block.find(
                "_dssp_struct_summary.",
                ["label_asym_id", "label_seq_id", "secondary_structure"],
            )
        except Exception as exc:
            return DsspOutcome(False, {}, f"parse_error: {type(exc).__name__}: {exc}")
        ss_map: dict[SsKey, SsCodes] = {}
        non_null = 0
        for row in table:
            chain_id, seq_id_str, ss_char = row[0], row[1], row[2]
            if seq_id_str in (".", "?"):
                continue
            try:
                seq_id = int(seq_id_str)
            except ValueError:
                continue
            ss3, ss8 = _ss_char_to_codes(ss_char)
            ss_map[(chain_id, seq_id)] = (ss3, ss8)
            if ss8 != SS8_NULL:
                non_null += 1
        if non_null == 0:
            return DsspOutcome(False, {}, "zero_non_null_ss")
        return DsspOutcome(True, ss_map, "")
