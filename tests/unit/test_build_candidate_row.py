from twistr.pipeline.curation.candidates import build_candidate_row
from twistr.config import Config


def _make_entry(
    rcsb_id: str,
    *,
    method: str = "X-RAY DIFFRACTION",
    resolution: float | None = 2.0,
    r_free: float | None = 0.22,
    em_resolution: float | None = None,
    n_polymer_entities: int = 1,
    polymer_entity_instance_count: int = 2,
    protein_entity_lengths: tuple[int, ...] = (250,),
    status: str = "REL",
) -> dict:
    refine = None
    if resolution is not None or r_free is not None:
        refine = [{"ls_d_res_high": resolution, "ls_R_factor_R_free": r_free}]
    em = None
    if em_resolution is not None:
        em = [{"resolution": em_resolution}]
    polymer_entities = [
        {
            "rcsb_id": f"{rcsb_id}_{i+1}",
            "entity_poly": {
                "type": "polypeptide(L)",
                "rcsb_sample_sequence_length": length,
                "pdbx_seq_one_letter_code_can": "A" * length,
            },
            "rcsb_entity_source_organism": None,
        }
        for i, length in enumerate(protein_entity_lengths)
    ]
    return {
        "rcsb_id": rcsb_id,
        "struct": {"title": f"test entry {rcsb_id}"},
        "exptl": [{"method": method}],
        "refine": refine,
        "em_3d_reconstruction": em,
        "rcsb_accession_info": {
            "status_code": status,
            "deposit_date": "2020-01-01T00:00:00Z",
            "initial_release_date": "2020-06-01T00:00:00Z",
        },
        "rcsb_entry_info": {
            "polymer_entity_count": n_polymer_entities,
            "polymer_entity_count_protein": n_polymer_entities,
            "polymer_entity_count_DNA": 0,
            "polymer_entity_count_RNA": 0,
            "polymer_composition": "homomeric protein",
            "resolution_combined": [resolution] if resolution is not None else None,
            "nonpolymer_entity_count": 0,
            "deposited_polymer_monomer_count": sum(protein_entity_lengths),
        },
        "assemblies": [
            {
                "rcsb_id": f"{rcsb_id}-1",
                "pdbx_struct_assembly": {"id": "1", "details": "author_defined_assembly"},
                "pdbx_struct_assembly_gen": [
                    {"assembly_id": "1", "oper_expression": "1", "asym_id_list": ["A"]}
                ],
                "rcsb_assembly_info": {
                    "polymer_entity_instance_count": polymer_entity_instance_count,
                    "polymer_entity_instance_count_protein": polymer_entity_instance_count,
                    "polymer_entity_instance_count_DNA": 0,
                    "polymer_entity_instance_count_RNA": 0,
                },
            }
        ],
        "polymer_entities": polymer_entities,
    }


def test_homomer_passes_chain_filter():
    """1 unique entity, 2 chain instances: the polymer_entity_count vs
    polymer_entity_instance_count trap. The filter must use instance_count."""
    entry = _make_entry(
        "1TIM",
        n_polymer_entities=1,
        polymer_entity_instance_count=2,
        protein_entity_lengths=(247,),
    )
    row = build_candidate_row(entry, obsoleted_from=None, cfg=Config())

    assert row.n_polymer_entities == 1
    assert row.n_instantiated_polymer_chains == 2
    assert row.passed_chains_filter, (
        "homomer (1 entity, 2 instances) failed passed_chains_filter; "
        "the filter is likely using polymer_entity_count instead of "
        "polymer_entity_instance_count"
    )
    assert row.passed_all_filters
    assert row.phase_a_drop_reason is None


def test_em_resolution_cascade():
    """EM entries have refine=None and resolution on em_3d_reconstruction.
    A bug that only reads refine.ls_d_res_high would leave resolution=None
    and drop every EM structure."""
    entry = _make_entry(
        "6VXX",
        method="ELECTRON MICROSCOPY",
        resolution=None,
        r_free=None,
        em_resolution=2.8,
        n_polymer_entities=1,
        polymer_entity_instance_count=3,
    )
    row = build_candidate_row(entry, obsoleted_from=None, cfg=Config())

    assert row.method == "ELECTRON MICROSCOPY"
    assert row.resolution == 2.8, (
        "EM entry must pick up resolution from em_3d_reconstruction "
        "when refine is absent"
    )
    assert row.passed_resolution_filter
    assert row.passed_rfree_filter, (
        "EM entries must skip the X-ray r_free filter entirely"
    )
    assert row.passed_all_filters


def test_xray_rfree_missing_keeps_and_tags():
    """Many valid X-ray structures have no r_free reported (e.g. 1BRS).
    Default config is keep_and_tag: they must pass the filter with
    rfree_missing=True tagged on the row."""
    entry = _make_entry(
        "1BRS",
        method="X-RAY DIFFRACTION",
        resolution=2.0,
        r_free=None,
        n_polymer_entities=2,
        polymer_entity_instance_count=2,
        protein_entity_lengths=(110, 89),
    )
    row = build_candidate_row(entry, obsoleted_from=None, cfg=Config())

    assert row.r_free is None
    assert row.rfree_missing is True
    assert row.passed_rfree_filter, (
        "X-ray entry with r_free=None must pass under default "
        "r_free_missing_action='keep_and_tag'; check that the "
        "comparator is not inverted"
    )
    assert row.passed_all_filters


def test_multi_filter_fail_produces_ordered_drop_reason():
    """A row that fails method, resolution, and chains must have
    phase_a_drop_reason 'filter:method,resolution,chains' in that exact
    order, matching the iteration order of FILTER_COLUMN_TO_NAME."""
    entry = _make_entry(
        "1G03",
        method="SOLUTION NMR",
        resolution=None,
        r_free=None,
        n_polymer_entities=1,
        polymer_entity_instance_count=1,
    )
    row = build_candidate_row(entry, obsoleted_from=None, cfg=Config())

    assert row.passed_method_filter is False
    assert row.passed_resolution_filter is False
    assert row.passed_chains_filter is False
    assert row.phase_a_drop_reason == "filter:method,resolution,chains", (
        f"expected ordered comma-joined failing filters, got "
        f"{row.phase_a_drop_reason!r}"
    )
