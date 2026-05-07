"""Generate a RosettaRemodel blueprint for a per-linker sub-pose.

The sub-pose contains the two anchor segments (rigid) plus Lk Ala
placeholder residues. The blueprint instructs Remodel to keep both
anchors fixed (`.`) and to rebuild the placeholder residues as a loop
(`L`) with sequence designed only over the loop-friendly amino-acid
whitelist (`PIKAA <set>`).
"""
from __future__ import annotations

from pathlib import Path

from .pose_builder import SubposeLayout


def write_blueprint(
    layout: SubposeLayout,
    aa_whitelist: str,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    up_lo, up_hi = layout.upstream_anchor
    for i, aa in enumerate(layout.upstream_aa, start=up_lo):
        lines.append(f"{i} {aa} .")

    link_lo, link_hi = layout.linker
    for i in range(link_lo, link_hi + 1):
        lines.append(f"{i} A L PIKAA {aa_whitelist}")

    dn_lo, dn_hi = layout.downstream_anchor
    for i, aa in enumerate(layout.downstream_aa, start=dn_lo):
        lines.append(f"{i} {aa} .")

    out_path.write_text('\n'.join(lines) + '\n')
    return out_path
