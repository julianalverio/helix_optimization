import pandas as pd

from twistr.curation.manifest import _compute_drop_reason


def _row(**fields) -> pd.Series:
    defaults = {
        "phase_a_drop_reason": None,
        "passed_all_filters": True,
        "file_present": True,
        "parse_ok": True,
        "parse_error": None,
        "max_protein_observed_fraction": 1.0,
    }
    defaults.update(fields)
    return pd.Series(defaults)


def test_precedence_order():
    """Phase A reason wins over parse state; download_missing wins over
    parse state; parse_error wins over observed_fraction; null when
    everything is clean."""
    phase_a_wins = _row(
        phase_a_drop_reason="filter:chains",
        parse_ok=False,
        parse_error="ValueError: bad",
        file_present=False,
    )
    assert _compute_drop_reason(phase_a_wins, 0.5) == "filter:chains", (
        "phase_a_drop_reason must take precedence over downstream failures"
    )

    download_missing = _row(file_present=False, parse_ok=False)
    assert _compute_drop_reason(download_missing, 0.5) == "download_missing", (
        "a Phase-A-passing record with no file on disk must report "
        "download_missing, not fall through to parse_error"
    )

    parse_error = _row(parse_ok=False, parse_error="ValueError: corrupt")
    assert _compute_drop_reason(parse_error, 0.5) == "parse_error:ValueError", (
        "parse_error prefix is 'parse_error:<ExceptionType>'; check the "
        "split on ':' is keeping only the type"
    )

    obs_filter = _row(max_protein_observed_fraction=0.1)
    assert _compute_drop_reason(obs_filter, 0.5) == "filter:observed_fraction"

    clean = _row()
    assert _compute_drop_reason(clean, 0.5) is None, (
        "a fully-passing record must return None, not any string"
    )


def test_nan_phase_a_drop_reason_does_not_trigger_phase_a_branch():
    """Regression guard: phase_a_drop_reason is an 'object' column in
    pandas; for passing rows the value is NaN (float), not None. Because
    bool(float('nan')) is True, a truthy check would incorrectly return
    NaN from the function. The implementation must use an 'isinstance str'
    check."""
    row = _row(phase_a_drop_reason=float("nan"))
    result = _compute_drop_reason(row, 0.5)
    assert result is None, (
        f"NaN in phase_a_drop_reason must not be treated as a phase-A "
        f"drop reason; got {result!r}"
    )

    row_obs_fail_with_nan_phase_a = _row(
        phase_a_drop_reason=float("nan"),
        max_protein_observed_fraction=0.1,
    )
    assert _compute_drop_reason(row_obs_fail_with_nan_phase_a, 0.5) == "filter:observed_fraction"
