from pathlib import Path

from twistr.pipeline.curation.obsolete import ObsoleteEntry, parse_obsolete, resolve_redirect


OBSOLETE_FIXTURE = """\
HEADER    OBSOLETE ENTRIES
OBSLTE     30-SEP-93 1AAA      2AAA
OBSLTE     15-JUN-01 1XYZ      2XYZ 3XYZ
OBSLTE     01-JAN-20 2AAA      3AAA
OBSLTE     01-JUN-22 9DED
"""


def test_parse_fixed_columns(tmp_path: Path):
    path = tmp_path / "obsolete.dat"
    path.write_text(OBSOLETE_FIXTURE)

    entries = parse_obsolete(path)

    assert set(entries.keys()) == {"1AAA", "1XYZ", "2AAA", "9DED"}
    assert entries["1AAA"].replacement_ids == ("2AAA",)
    assert entries["1XYZ"].replacement_ids == ("2XYZ", "3XYZ"), (
        "multi-replacement line must capture all successor IDs"
    )
    assert entries["9DED"].replacement_ids == (), (
        "trailing-whitespace no-replacement line must parse to an empty tuple, "
        "not incorrectly attach the next line's content"
    )


def test_resolve_redirect_chain_and_cycle(tmp_path: Path):
    path = tmp_path / "obsolete.dat"
    path.write_text(OBSOLETE_FIXTURE)
    entries = parse_obsolete(path)

    assert resolve_redirect("1AAA", entries) == "3AAA", (
        "resolve_redirect must follow the 1AAA -> 2AAA -> 3AAA chain to its terminal replacement"
    )
    assert resolve_redirect("1XYZ", entries) == "2XYZ", (
        "multi-replacement entries resolve to the first listed replacement"
    )
    assert resolve_redirect("9DED", entries) is None, (
        "dead-end entries (no replacement) must return None, not fall through"
    )
    assert resolve_redirect("NEVR", entries) == "NEVR", (
        "never-obsoleted IDs pass through unchanged"
    )

    cycle_fixture = {
        "1AAA": ObsoleteEntry("1AAA", ("2AAA",), None),
        "2AAA": ObsoleteEntry("2AAA", ("1AAA",), None),
    }
    assert resolve_redirect("1AAA", cycle_fixture) is None, (
        "cycles must return None; an infinite loop would hang the pipeline"
    )
