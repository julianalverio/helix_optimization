import gzip
from pathlib import Path

import pytest

from twistr.curation import verify

FIXTURE_MMCIF = Path(__file__).parent.parent / "fixtures" / "mmCIF"


def test_parse_1brs_observed_fraction():
    """gemmi traversal: every chain in 1BRS has observed_fraction close to
    1.0 and a non-empty canonical sequence. Catches regressions in chain
    iteration, entity lookup, SEQRES length, and the heterogeneity-stripped
    one-letter conversion."""
    result = verify.parse_structure(FIXTURE_MMCIF / "1BRS.cif.gz")

    assert result.parse_ok, f"1BRS failed to parse: {result.parse_error}"
    assert result.method == "X-RAY DIFFRACTION"
    assert result.n_protein_chains == 6, (
        f"1BRS has 3 barnase + 3 barstar chains in the AU; got "
        f"{result.n_protein_chains}"
    )
    assert result.max_protein_observed_fraction is not None
    assert result.max_protein_observed_fraction <= 1.0 + 1e-9
    assert result.min_protein_observed_fraction >= 0.9, (
        f"1BRS chains are well-resolved; observed_fraction lower bound "
        f"of 0.9 is safely above known values, got "
        f"{result.min_protein_observed_fraction}"
    )

    for chain in result.chains:
        assert chain.seqres_length > 0, (
            f"chain {chain.chain_id}: SEQRES length must be > 0; "
            f"full_sequence extraction is broken"
        )
        assert 0.0 <= chain.observed_fraction <= 1.0 + 1e-9, (
            f"chain {chain.chain_id}: observed_fraction out of range: "
            f"{chain.observed_fraction}"
        )
        if chain.chain_type == "protein":
            assert chain.canonical_sequence, (
                f"chain {chain.chain_id}: canonical_sequence empty; "
                f"likely a heterogeneity-marker parse failure "
                f"(ALA;SER or ALA,SER in full_sequence)"
            )
            assert len(chain.canonical_sequence) == chain.seqres_length, (
                f"chain {chain.chain_id}: canonical_sequence length "
                f"{len(chain.canonical_sequence)} != seqres_length "
                f"{chain.seqres_length}; one-letter conversion lost residues"
            )


def test_parse_corrupt_file_returns_parse_error(tmp_path: Path):
    """Error isolation regression guard: a truncated / non-mmCIF .cif.gz
    must return parse_ok=False with parse_error populated, NOT raise.
    A regression here would crash Phase C on the first bad file."""
    bad = tmp_path / "xxxx.cif.gz"
    with gzip.open(bad, "wb") as f:
        f.write(b"not a real mmcif file\n")

    result = verify.parse_structure(bad)

    assert result.parse_ok is False
    assert result.parse_error is not None, (
        "corrupt file must populate parse_error; silent empty errors "
        "break the drop_reason precedence chain"
    )
    assert result.pdb_id == "XXXX"
