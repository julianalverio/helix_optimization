import pytest

from twistr.pipeline.curation import rcsb

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def session():
    return rcsb.build_session()


@pytest.fixture(scope="module")
def metadata(session, tmp_path_factory):
    cache = tmp_path_factory.mktemp("phase_a_cache")
    ids = ["1HH6", "1BRS", "2REB", "1TIM", "6VXX", "1AON", "1A3N"]
    entries, failed = rcsb.fetch_metadata(session, ids, cache_dir=cache)
    assert failed == []
    return {e["rcsb_id"].upper(): e for e in entries}


def test_1brs_is_xray(metadata):
    methods = [m["method"] for m in metadata["1BRS"]["exptl"]]
    assert "X-RAY DIFFRACTION" in methods


def test_1tim_is_homodimer(metadata):
    assemblies = metadata["1TIM"]["assemblies"]
    info = assemblies[0]["rcsb_assembly_info"]
    assert info["polymer_entity_instance_count"] == 2
    assert metadata["1TIM"]["rcsb_entry_info"]["polymer_entity_count"] == 1


def test_2reb_is_monomeric_in_biological_assembly(metadata):
    assemblies = metadata["2REB"]["assemblies"]
    info = assemblies[0]["rcsb_assembly_info"]
    assert info["polymer_entity_instance_count"] == 1


def test_1a3n_has_four_chains(metadata):
    assemblies = metadata["1A3N"]["assemblies"]
    info = assemblies[0]["rcsb_assembly_info"]
    assert info["polymer_entity_instance_count"] == 4


def test_6vxx_is_large_assembly(metadata):
    assemblies = metadata["6VXX"]["assemblies"]
    info = assemblies[0]["rcsb_assembly_info"]
    assert info["polymer_entity_instance_count"] >= 3


def test_fetch_all_released_ids_returns_many(session):
    ids = rcsb.fetch_all_released_ids(session)
    assert len(ids) > 100_000


def test_fetch_metadata_handles_nonexistent_id(session, tmp_path):
    entries, failed = rcsb.fetch_metadata(session, ["1BRS", "9ZZZ"], cache_dir=tmp_path)
    returned_ids = {e["rcsb_id"].upper() for e in entries}
    assert "1BRS" in returned_ids
    assert "9ZZZ" not in returned_ids
