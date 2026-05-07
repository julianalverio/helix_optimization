"""Standard 4-class biochemical grouping used to color epitope patch sticks."""
from __future__ import annotations

HYDROPHOBIC: frozenset[str] = frozenset("AVLIMFWYPCG")
NEGATIVE: frozenset[str] = frozenset("DE")
POSITIVE: frozenset[str] = frozenset("KRH")
POLAR: frozenset[str] = frozenset("STNQ")

CATEGORIES = ("hydrophobic", "negative", "positive", "polar")


def classify(one_letter: str) -> str | None:
    aa = one_letter.upper()
    if aa in HYDROPHOBIC:
        return "hydrophobic"
    if aa in NEGATIVE:
        return "negative"
    if aa in POSITIVE:
        return "positive"
    if aa in POLAR:
        return "polar"
    return None
