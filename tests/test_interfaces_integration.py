import pytest

from twistr.pipeline.curation import interfaces, rcsb

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def session():
    return rcsb.build_session()


def test_1brs_has_single_interface(session):
    plan = interfaces.fetch_unique_interfaces(session, "1BRS", "1")
    assert len(plan) == 1
    assert {plan[0].entity_id_1, plan[0].entity_id_2} == {"1", "2"}


def test_1aon_interfaces_cover_all_entity_pairs(session):
    plan = interfaces.fetch_unique_interfaces(session, "1AON", "1")
    pairs = {tuple(sorted((p.entity_id_1, p.entity_id_2))) for p in plan}
    assert ("1", "1") in pairs
    assert ("1", "2") in pairs
    assert ("2", "2") in pairs
    assert 3 <= len(plan) <= 30
