import json
from pathlib import Path

from twistr.pipeline.curation import interfaces

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "rcsb_interfaces"


def _load_1aon():
    assembly = json.loads((FIXTURE_DIR / "1AON_1_assembly.json").read_text())
    interface_ids = (assembly.get("rcsb_assembly_container_identifiers") or {}).get("interface_ids") or []
    interfaces_by_id = {
        str(iid): json.loads((FIXTURE_DIR / f"1AON_1_{iid}.json").read_text())
        for iid in interface_ids
    }
    return assembly, interfaces_by_id


def test_1aon_unique_interfaces_dedupe():
    """1AON (GroEL-GroES, 21 chains, 2 entities) has 42 raw interfaces
    from RCSB. The dedupe heuristic must:
      - produce a plan significantly smaller than the raw count
      - cover every entity-pair present in the raw data (GroEL-GroEL,
        GroEL-GroES, GroES-GroES)
      - have no collisions in the internal dedupe key

    Under-dedupe would flood Module 2 with redundant interface
    expansions; over-dedupe would silently drop real distinct contacts.
    """
    assembly, interfaces_by_id = _load_1aon()
    raw_count = len(interfaces_by_id)
    assert raw_count > 20, (
        f"fixture sanity: 1AON should have many raw interfaces (~42), got {raw_count}"
    )

    plan = interfaces.dedupe_from_responses(assembly, interfaces_by_id)

    assert 3 <= len(plan) <= 20, (
        f"1AON plan size {len(plan)} out of reasonable bounds [3, 20]. "
        f"Raw interfaces: {raw_count}. Likely a regression in _dedupe_key "
        f"buckets (under-dedupe) or dedupe over-collapsed distinct contacts."
    )
    assert len(plan) < raw_count, (
        "dedupe produced plan as large as raw input; the dedupe check "
        "is no-op"
    )

    pairs_seen = {
        tuple(sorted((p.entity_id_1 or "", p.entity_id_2 or "")))
        for p in plan
    }
    for required in [("1", "1"), ("1", "2"), ("2", "2")]:
        assert required in pairs_seen, (
            f"entity-pair {required} missing from deduped plan; "
            f"pairs present: {sorted(pairs_seen)}. Dedupe key has "
            f"likely collapsed distinct entity-pair interfaces."
        )

    keys = [
        interfaces._dedupe_key(
            (p.entity_id_1, p.entity_id_2),
            p.interface_area,
            interfaces._extract_num_residues(interfaces_by_id[p.interface_id]),
        )
        for p in plan
    ]
    assert len(keys) == len(set(keys)), (
        "duplicate dedupe keys survived; the set-membership check in "
        "dedupe_from_responses is not functioning"
    )
