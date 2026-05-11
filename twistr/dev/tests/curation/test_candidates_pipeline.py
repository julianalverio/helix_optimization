import pandas as pd
import pytest

from twistr.curation.config import Config
from twistr.curation import candidates, rcsb
from twistr.curation.obsolete import fetch_obsolete, parse_obsolete

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def cfg():
    return Config()


@pytest.fixture(scope="module")
def phase_a_df(tmp_path_factory, cfg):
    root = tmp_path_factory.mktemp("data_root")
    session = rcsb.build_session()
    aux = root / "aux"
    aux.mkdir()
    obs_path = aux / "obsolete.dat"
    fetch_obsolete(session, obs_path)
    obsolete_map = parse_obsolete(obs_path)

    ids = ["1HH6", "1BRS", "2REB", "1TIM", "6VXX", "1AON", "1A3N", "1G03"]
    resolved, redirect_map, _dropped = candidates.resolve_candidate_ids(ids, obsolete_map)

    cache = root / "cache"
    metadata, _failed = rcsb.fetch_metadata(session, resolved, cache_dir=cache)
    entries_by_id = {e["rcsb_id"].upper(): e for e in metadata}

    rows = [
        candidates.build_candidate_row(entries_by_id[pid], redirect_map.get(pid), cfg)
        for pid in resolved
        if pid in entries_by_id
    ]
    return pd.DataFrame([r.__dict__ for r in rows]).set_index("pdb_id")


def test_1brs_passes(phase_a_df):
    row = phase_a_df.loc["1BRS"]
    assert row["passed_all_filters"]


def test_1tim_homodimer_passes_despite_single_entity(phase_a_df):
    row = phase_a_df.loc["1TIM"]
    assert row["n_polymer_entities"] == 1
    assert row["n_instantiated_polymer_chains"] >= 2
    assert row["passed_chains_filter"]
    assert row["passed_all_filters"]


def test_2reb_monomeric_assembly_fails_chain_filter(phase_a_df):
    row = phase_a_df.loc["2REB"]
    assert row["n_polymer_entities"] == 1
    assert row["n_instantiated_polymer_chains"] == 1
    assert not row["passed_chains_filter"]


def test_6vxx_passes(phase_a_df):
    row = phase_a_df.loc["6VXX"]
    assert row["passed_all_filters"]
    assert row["method"] == "ELECTRON MICROSCOPY"


def test_1aon_is_large_assembly(phase_a_df):
    row = phase_a_df.loc["1AON"]
    assert row["large_assembly"]
    assert row["n_instantiated_polymer_chains"] >= 20


def test_1a3n_passes(phase_a_df):
    row = phase_a_df.loc["1A3N"]
    assert row["passed_all_filters"]


def test_nmr_entry_is_rejected(phase_a_df):
    row = phase_a_df.loc["1G03"]
    assert not row["passed_method_filter"]
    assert not row["passed_all_filters"]
    assert row["phase_a_drop_reason"].startswith("filter:")
    assert "method" in row["phase_a_drop_reason"]


def test_bogus_id_lands_in_audit_as_metadata_missing(tmp_path):
    from twistr.curation import candidates
    from twistr.curation.config import Config, snapshot_now
    from twistr.curation import paths as mpaths

    root = tmp_path
    (root / "aux").mkdir()
    (root / "manifests").mkdir()
    (root / "pdb").mkdir()

    import twistr.curation.candidates as cand_mod
    original_select = cand_mod.select_input_ids
    cand_mod.select_input_ids = lambda session, full_scale: ["1BRS", "9ZZZ"]
    try:
        out = candidates.run_phase_a(Config(), root, full_scale=False, snapshot_date=snapshot_now())
    finally:
        cand_mod.select_input_ids = original_select

    import pandas as pd
    df = pd.read_parquet(out).set_index("pdb_id")
    assert "9ZZZ" in df.index
    assert df.loc["9ZZZ", "phase_a_drop_reason"] == "metadata_missing"
    assert not df.loc["9ZZZ", "passed_all_filters"]
    assert df.loc["1BRS", "passed_all_filters"]
