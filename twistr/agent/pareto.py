"""Pareto frontier over (shape complementarity, electrostatic
complementarity, buried surface area). All three objectives are
*maximized*. A design is non-dominated iff no other design is ≥ on all
three objectives and > on at least one."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Design:
    """A scored design (sequence + metrics) on the search trajectory."""
    id: int
    parent_id: int | None
    mutations: list[tuple[str, int, str, str]]
    """(chain_id, residue_number, from_aa, to_aa) tuples accumulated
    from the WT along this design's lineage. Empty list = WT."""

    helix_sequences: dict[str, str]
    """Current 1-letter sequence keyed by helix chain ID."""

    sc: float
    """Shape complementarity, [-1, +1]; higher is better."""

    ec: float
    """Electrostatic complementarity, [-1, +1]; higher is better."""

    bsa: float
    """Buried surface area in Å²; higher is better."""

    iteration: int = 0
    """Agent-loop step at which this design was scored."""

    notes: str = ""
    """Free-form annotation from the agent (e.g. reasoning summary)."""


def _ge(a: float, b: float) -> bool:
    """`a` ≥ `b` for objective-comparison purposes, treating NaN as the
    worst possible value (NaN < anything that's not NaN)."""
    if math.isnan(a):
        return math.isnan(b)
    if math.isnan(b):
        return True
    return a >= b


def _gt(a: float, b: float) -> bool:
    if math.isnan(a):
        return False
    if math.isnan(b):
        return True
    return a > b


class ParetoFrontier:
    """Maintains the non-dominated set under (SC, EC, BSA) — all three
    maximized. Insertion is O(|frontier|); for the scale we run (a few
    hundred designs per agent session), this is fine and the resulting
    `designs` list is always exactly the current Pareto-optimal set."""

    def __init__(self) -> None:
        self._designs: list[Design] = []

    def add(self, design: Design) -> bool:
        """Insert a design. Returns True if it ends up on the frontier;
        False if it was dominated by an existing member (and discarded)."""
        for existing in self._designs:
            if self._dominates(existing, design):
                return False
        # New design is non-dominated. Evict any existing member it
        # dominates, then admit.
        self._designs = [d for d in self._designs if not self._dominates(design, d)]
        self._designs.append(design)
        return True

    @staticmethod
    def _dominates(a: Design, b: Design) -> bool:
        ge_all = _ge(a.sc, b.sc) and _ge(a.ec, b.ec) and _ge(a.bsa, b.bsa)
        gt_any = _gt(a.sc, b.sc) or _gt(a.ec, b.ec) or _gt(a.bsa, b.bsa)
        return ge_all and gt_any

    @property
    def designs(self) -> list[Design]:
        return list(self._designs)

    def __len__(self) -> int:
        return len(self._designs)

    def best_by(self, metric: str) -> Design | None:
        """Frontier member maximizing the named metric. Useful for the
        agent's prompt when it wants 'the SC-extremum design'."""
        if not self._designs:
            return None
        return max(self._designs, key=lambda d: getattr(d, metric))

    def to_jsonl(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for design in self._designs:
                f.write(json.dumps(asdict(design)) + "\n")

    @classmethod
    def from_jsonl(cls, path: Path) -> "ParetoFrontier":
        frontier = cls()
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                payload["mutations"] = [tuple(m) for m in payload["mutations"]]
                frontier._designs.append(Design(**payload))
        return frontier
