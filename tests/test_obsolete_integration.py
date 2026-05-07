import pytest

from twistr.pipeline.curation import obsolete, rcsb

pytestmark = pytest.mark.network


def test_fetch_and_parse_obsolete(tmp_path):
    session = rcsb.build_session()
    dest = tmp_path / "obsolete.dat"
    obsolete.fetch_obsolete(session, dest)
    assert dest.exists()
    assert dest.stat().st_size > 0
    entries = obsolete.parse_obsolete(dest)
    assert len(entries) > 1000


def test_resolve_redirect_follows_chain(tmp_path):
    session = rcsb.build_session()
    dest = tmp_path / "obsolete.dat"
    obsolete.fetch_obsolete(session, dest)
    entries = obsolete.parse_obsolete(dest)
    obsoleted = [eid for eid, e in entries.items() if e.replacement_ids][:1]
    assert obsoleted, "expected at least one obsoleted entry with replacement"
    eid = obsoleted[0]
    replacement = obsolete.resolve_redirect(eid, entries)
    assert replacement is not None
    assert replacement != eid
