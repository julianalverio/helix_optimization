from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests

OBSOLETE_URL = "https://files.wwpdb.org/pub/pdb/data/status/obsolete.dat"


@dataclass(frozen=True)
class ObsoleteEntry:
    obsoleted_id: str
    replacement_ids: tuple[str, ...]
    obsoletion_date: str | None


def fetch_obsolete(session: requests.Session, dest: Path) -> Path:
    response = session.get(OBSOLETE_URL, timeout=60)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(response.content)
    tmp.replace(dest)
    return dest


def parse_obsolete(path: Path) -> dict[str, ObsoleteEntry]:
    entries: dict[str, ObsoleteEntry] = {}
    with path.open() as f:
        for line in f:
            if not line.startswith("OBSLTE"):
                continue
            tokens = line.split()
            if len(tokens) < 3:
                continue
            date_token = tokens[1]
            obsoleted = tokens[2].upper()
            replacements = tuple(tok.upper() for tok in tokens[3:])
            obsoletion_date = _iso_date(date_token)
            entries[obsoleted] = ObsoleteEntry(obsoleted, replacements, obsoletion_date)
    return entries


def _iso_date(token: str) -> str | None:
    try:
        return datetime.strptime(token, "%d-%b-%y").date().isoformat()
    except ValueError:
        return None


def resolve_redirect(pdb_id: str, entries: dict[str, ObsoleteEntry]) -> str | None:
    """Follow the obsolete chain. Returns the current non-obsolete replacement,
    or None if the chain dead-ends (no replacement or cycle)."""
    seen: set[str] = set()
    current = pdb_id.upper()
    while current in entries:
        if current in seen:
            return None
        seen.add(current)
        reps = entries[current].replacement_ids
        if not reps:
            return None
        current = reps[0]
    return current
