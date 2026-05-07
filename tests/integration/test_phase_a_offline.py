import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from twistr import paths
from twistr.config import Config, config_hash
from twistr.pipeline.curation import candidates

FIXTURE_GRAPHQL = Path(__file__).parent.parent / "fixtures" / "rcsb_graphql"
TEST_IDS = ["1BRS", "1TIM", "1G03", "1HH6", "6VXX", "1A3N"]


def _seed_graphql_cache(data_root: Path, cfg: Config) -> None:
    target = paths.cache_dir(data_root, config_hash(cfg), "phase_a")
    target.mkdir(parents=True, exist_ok=True)
    for src in FIXTURE_GRAPHQL.glob("*.jsonl"):
        shutil.copy(src, target / src.name)


def test_phase_a_offline_end_to_end(tmp_path: Path):
    """End-to-end Phase A against a pre-seeded GraphQL cache. Asserts the
    integration-level behavior of build_candidate_row + drop_reason +
    stub-row emission on a realistic dev subset, with zero network access.

    Every assertion targets a specific class of regression:
      - homomer (1TIM): instance-count trap
      - NMR (1G03): method filter + multi-reason drop_reason
      - high r_free (1HH6): rfree filter direction
      - EM (6VXX): resolution cascade from em_3d_reconstruction
      - X-ray with r_free=None (1BRS): keep_and_tag behavior
    """
    cfg = Config()
    paths.ensure_dirs(tmp_path)
    _seed_graphql_cache(tmp_path, cfg)

    snapshot = datetime(2026, 4, 22, tzinfo=timezone.utc)
    out_path = candidates.run_phase_a_on_ids(
        cfg=cfg,
        data_root_path=tmp_path,
        input_ids=TEST_IDS,
        snapshot_date=snapshot,
        obsolete_map={},
    )

    df = pd.read_parquet(out_path).set_index("pdb_id")
    assert set(df.index) >= set(TEST_IDS), (
        f"candidates.parquet missing some test IDs. present: {sorted(df.index)}"
    )

    tim = df.loc["1TIM"]
    assert tim.n_polymer_entities == 1
    assert tim.n_instantiated_polymer_chains == 2
    assert tim.passed_all_filters, (
        "1TIM is the canonical homodimer regression case; "
        "a failure here means polymer_entity_count is being used "
        "as the chain filter"
    )
    assert pd.isna(tim.phase_a_drop_reason), (
        f"passing row should have no phase_a_drop_reason, got {tim.phase_a_drop_reason!r}"
    )

    g03 = df.loc["1G03"]
    assert not g03.passed_method_filter
    assert not g03.passed_all_filters
    assert g03.phase_a_drop_reason == "filter:method,resolution,chains", (
        f"1G03 (NMR, no resolution, 1 chain) expected drop_reason "
        f"'filter:method,resolution,chains', got {g03.phase_a_drop_reason!r}"
    )

    hh6 = df.loc["1HH6"]
    assert hh6.r_free is not None
    assert hh6.r_free > 0.30
    assert hh6.phase_a_drop_reason == "filter:rfree", (
        f"1HH6 has r_free=0.338 which exceeds the 0.30 threshold; "
        f"expected drop_reason 'filter:rfree', got "
        f"{hh6.phase_a_drop_reason!r}. Check the comparator direction."
    )

    vxx = df.loc["6VXX"]
    assert vxx.method == "ELECTRON MICROSCOPY"
    assert vxx.resolution is not None, (
        "6VXX resolution must be picked up from em_3d_reconstruction; "
        "a None here means the EM resolution cascade is broken"
    )
    assert vxx.resolution <= cfg.resolution_max_em
    assert vxx.passed_all_filters

    brs = df.loc["1BRS"]
    assert pd.isna(brs.r_free), (
        f"1BRS has no reported r_free; expected NaN, got {brs.r_free!r}"
    )
    assert bool(brs.rfree_missing), (
        "1BRS has no reported r_free; rfree_missing tag must be True"
    )
    assert brs.passed_rfree_filter, (
        "default config keeps and tags X-ray entries missing r_free; "
        "if this fails, the keep_and_tag branch is inverted"
    )
    assert brs.passed_all_filters
