# Module 1 Minimal Test Implementation Plan (for AI Agent Execution)

## Goal

Implement a minimal test suite for Module 1 of the PDB protein-protein interaction pipeline. The suite targets the three areas where custom logic is most likely to be misimplemented in ways that would silently corrupt data for Module 2:

1. Homo-oligomer chain counting (the `polymer_entity_count` vs `polymer_entity_instance_count` trap)
2. Unique interface planning for large assemblies (custom operator-expression parsing and deduplication)
3. Chain-ID convention consistency (`label_asym_id` vs `auth_asym_id` used consistently across all outputs)

This is intentionally a small suite. Mature libraries handle the rest of Module 1's logic; downstream modules or the Step 20 summary report will surface other classes of bugs.

## Design Principles

- **Small and fast.** The full suite should run in under 30 seconds.
- **Offline.** No network access. All fixtures are committed to the repo.
- **Deterministic.** No flakiness; no random sampling.
- **Fails loudly.** Every assertion has a message that names the offending value.
- **Session-scoped pipeline run.** The pipeline runs once over the fixture set and all tests consume the results.

## Repo Layout

The agent must create the following directory structure:

```
tests/
  conftest.py
  fixtures/
    mmCIF/
      2reb.cif.gz      # RecA homohexamer — homomer chain-counting test
      1aon.cif.gz      # GroEL-GroES, 21 chains — large-assembly plan test
      1a3n.cif.gz      # Hemoglobin — chain-ID convention test
    expected/
      2reb.json
      1aon.json
      1a3n.json
  test_module1.py
```

## Step 1 — Fixture Preparation

### 1a. Download and commit fixture mmCIFs

Run these commands to fetch the fixtures:

```bash
mkdir -p tests/fixtures/mmCIF
for id in 2reb 1aon 1a3n; do
  curl -s "https://files.rcsb.org/download/${id}.cif.gz" \
    -o "tests/fixtures/mmCIF/${id}.cif.gz"
done
```

Verify each file is parseable with the pipeline's primary parser (gemmi) before committing. Each file should be under 2 MB; the three fixtures combined should be under 5 MB.

### 1b. Create expected-value JSON files

For each fixture, hand-verify the expected values by running the pipeline once and inspecting the output. Then write the expected JSON files. The agent must verify these values are correct against the actual fixtures before writing tests. If a fixture's current output differs from these values, investigate: either the fixture is wrong for the test's purpose, or the expected JSON needs updating, or there is a real bug.

Write `tests/fixtures/expected/2reb.json` with the following contents:

```json
{
  "pdb_id": "2REB",
  "passed_all_filters": true,
  "n_polymer_entities": 1,
  "n_instantiated_polymer_chains_min": 2,
  "large_assembly": false
}
```

Write `tests/fixtures/expected/1aon.json` with the following contents:

```json
{
  "pdb_id": "1AON",
  "passed_all_filters": true,
  "large_assembly": true,
  "unique_interface_plan_max_length": 10
}
```

Write `tests/fixtures/expected/1a3n.json` with the following contents:

```json
{
  "pdb_id": "1A3N",
  "passed_all_filters": true,
  "n_polymer_entities": 2,
  "n_instantiated_polymer_chains": 4
}
```

## Step 2 — Implement `tests/conftest.py`

Write the following to `tests/conftest.py`. If Module 1's entrypoint signature differs from `run_pipeline(input_mmcif_dir, output_dir, pdb_id_list, skip_download)`, the agent should adapt the `pipeline_outputs` fixture accordingly. The contract is: given a directory of mmCIF files and a list of PDB IDs, produce `module1_manifest.parquet` and `chain_sequences.parquet` in the output directory.

```python
import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
MMCIF_DIR = FIXTURE_DIR / "mmCIF"
EXPECTED_DIR = FIXTURE_DIR / "expected"


@pytest.fixture(scope="session")
def mmcif_dir():
    return MMCIF_DIR


@pytest.fixture(scope="session")
def fixture_pdb_ids():
    return ["2REB", "1AON", "1A3N"]


@pytest.fixture(scope="session")
def expected():
    """Returns a dict keyed by uppercase PDB ID."""
    out = {}
    for path in EXPECTED_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        out[data["pdb_id"].upper()] = data
    return out


@pytest.fixture(scope="session")
def pipeline_outputs(tmp_path_factory, fixture_pdb_ids):
    """Runs Module 1 once over the fixture set and returns output tables."""
    out_dir = tmp_path_factory.mktemp("module1_output")

    # Adapt this import and call to Module 1's actual entrypoint.
    from pipeline.module1 import run_pipeline
    run_pipeline(
        input_mmcif_dir=MMCIF_DIR,
        output_dir=out_dir,
        pdb_id_list=fixture_pdb_ids,
        skip_download=True,
    )

    manifest = pd.read_parquet(out_dir / "module1_manifest.parquet").set_index("pdb_id")
    chain_sequences = pd.read_parquet(out_dir / "chain_sequences.parquet")

    return {
        "manifest": manifest,
        "chain_sequences": chain_sequences,
        "output_dir": out_dir,
    }


@pytest.fixture(scope="session")
def manifest(pipeline_outputs):
    return pipeline_outputs["manifest"]


@pytest.fixture(scope="session")
def chain_sequences(pipeline_outputs):
    return pipeline_outputs["chain_sequences"]
```

## Step 3 — Implement `tests/test_module1.py`

Write the following to `tests/test_module1.py`. Implement exactly three test functions. Each has a docstring stating what class of bug it protects against.

```python
import gemmi


def test_homomer_passes_chain_filter(manifest, expected):
    """Protects against the polymer_entity_count vs instance_count trap.

    A homo-oligomer has one unique entity but multiple chain instances in
    the biological assembly. A naive filter using polymer_entity_count
    would incorrectly drop it. This test asserts that the filter uses
    polymer_entity_instance_count so homomers pass.

    Failure here means roughly 30-40% of multi-chain PDB entries (all
    homo-oligomers) are being silently excluded from the dataset.
    """
    pdb_id = "2REB"
    exp = expected[pdb_id]

    assert pdb_id in manifest.index, (
        f"{pdb_id} (RecA homohexamer) is missing from the manifest. "
        f"Likely cause: chain-count filter uses polymer_entity_count "
        f"instead of polymer_entity_instance_count, dropping all homomers."
    )

    row = manifest.loc[pdb_id]
    assert row.passed_all_filters, (
        f"{pdb_id} failed filters. drop_reason={getattr(row, 'drop_reason', 'N/A')}. "
        f"Homomers must pass the chain-count filter."
    )
    assert row.n_polymer_entities == exp["n_polymer_entities"], (
        f"Expected {exp['n_polymer_entities']} unique entity for {pdb_id}, "
        f"got {row.n_polymer_entities}"
    )
    assert row.n_instantiated_polymer_chains >= exp["n_instantiated_polymer_chains_min"], (
        f"{pdb_id} has n_instantiated_polymer_chains="
        f"{row.n_instantiated_polymer_chains}, expected >= "
        f"{exp['n_instantiated_polymer_chains_min']}. "
        f"Likely cause: assembly not expanded correctly, or chain counting "
        f"uses asymmetric unit instead of biological assembly."
    )


def test_large_assembly_plan_is_valid(manifest, expected, mmcif_dir):
    """Protects against bugs in unique-interface planning for large
    assemblies — the one area of Module 1 that is genuinely novel code.

    Checks four invariants on the unique_interface_plan:
    1. It is non-null and non-empty for large assemblies.
    2. Every referenced operator ID exists in the source mmCIF.
    3. The plan is deduplicated (no two entries with the same interface_hash).
    4. The plan length is bounded (sanity check on dedup).

    Failure here means the interface plan for large assemblies is wrong,
    which would cause Module 2 to either skip real interfaces or
    generate duplicate / invalid training examples from capsids and
    other large complexes.
    """
    pdb_id = "1AON"
    exp = expected[pdb_id]
    row = manifest.loc[pdb_id]

    # Invariant 1: plan exists and is non-empty
    assert row.large_assembly, (
        f"{pdb_id} should be flagged large_assembly=True "
        f"(it has 21 chains). Check large_assembly_chain_threshold."
    )
    plan = row.unique_interface_plan
    assert plan is not None and len(plan) > 0, (
        f"{pdb_id} is large_assembly but unique_interface_plan is "
        f"null or empty. Interface planner failed silently."
    )

    # Invariant 2: every referenced operator exists in the source mmCIF
    structure = gemmi.read_structure(str(mmcif_dir / "1aon.cif.gz"))
    valid_op_names = _collect_operator_names(structure)
    for i, entry in enumerate(plan):
        assert entry["operator_1"] in valid_op_names, (
            f"{pdb_id} plan entry {i}: operator_1='{entry['operator_1']}' "
            f"is not in source mmCIF operator list. "
            f"Likely cause: operator-expression parser mishandled the "
            f"expression and produced invalid operator IDs."
        )
        assert entry["operator_2"] in valid_op_names, (
            f"{pdb_id} plan entry {i}: operator_2='{entry['operator_2']}' "
            f"is not in source mmCIF operator list."
        )

    # Invariant 3: plan is deduplicated by interface_hash
    hashes = [entry["interface_hash"] for entry in plan]
    assert len(hashes) == len(set(hashes)), (
        f"{pdb_id} unique_interface_plan contains duplicate interface_hash "
        f"values. Deduplication is broken. "
        f"Hashes: {hashes}"
    )

    # Invariant 4: plan length is bounded. A 21-chain assembly with proper
    # symmetry dedup should have < 10 unique interfaces. More than that
    # means dedup is failing.
    assert len(plan) <= exp["unique_interface_plan_max_length"], (
        f"{pdb_id} unique_interface_plan has {len(plan)} entries, "
        f"expected <= {exp['unique_interface_plan_max_length']}. "
        f"Likely cause: symmetry-equivalent interfaces are not being "
        f"deduplicated."
    )


def test_chain_ids_use_label_asym_id_consistently(manifest, chain_sequences, mmcif_dir):
    """Protects against mixing label_asym_id and auth_asym_id across the
    manifest, chain_sequences table, and interface plans.

    Module 1's documented convention is label_asym_id everywhere. This
    test picks a fixture where author and label chain IDs differ and
    asserts that chain_sequences uses label IDs.

    Failure here means downstream joins (manifest to chain_sequences,
    chain_sequences to interface plan) will produce empty or wrong
    results, silently breaking Module 2.
    """
    pdb_id = "1A3N"
    structure = gemmi.read_structure(str(mmcif_dir / "1a3n.cif.gz"))

    # Collect label_asym_id values for all polymer chains in the AU.
    label_ids = {chain.name for chain in structure[0]}

    # Chain-IDs used in chain_sequences.parquet for this entry.
    seq_chain_ids = set(
        chain_sequences[chain_sequences.pdb_id == pdb_id].chain_id.astype(str)
    )

    assert seq_chain_ids, (
        f"No chain sequences recorded for {pdb_id}. "
        f"Either sequence extraction failed or pdb_id casing mismatches."
    )

    unexpected = seq_chain_ids - label_ids
    assert not unexpected, (
        f"{pdb_id}: chain_sequences contains chain IDs "
        f"{unexpected} that are not valid label_asym_id values. "
        f"Valid label_asym_ids are {label_ids}. "
        f"Likely cause: sequence extractor is using auth_asym_id instead "
        f"of label_asym_id. This will break joins with the manifest and "
        f"interface plan."
    )


def _collect_operator_names(structure):
    """Returns the set of operator names defined in the mmCIF's
    pdbx_struct_oper_list. Adjust if the pipeline stores operator IDs
    differently (e.g., as strings vs integers)."""
    names = set()
    for assembly in structure.assemblies:
        for gen in assembly.generators:
            for op in gen.operators:
                names.add(op.name)
    return names
```

## Step 4 — Run the Tests

Run the suite:

```bash
pytest tests/ -v
```

Expected: three tests pass in under 30 seconds.

## Step 5 — Bug-Injection Verification

For each test, prove it actually catches its target bug by temporarily introducing the bug into the pipeline and confirming the test fails with a clear message. Revert each injection after verification.

Perform these four injections one at a time:

1. **For `test_homomer_passes_chain_filter`:** In the chain-count filter, replace `polymer_entity_instance_count` with `polymer_entity_count`. Expected result: test fails because `2REB` is missing or has `passed_all_filters=False`.

2. **For `test_large_assembly_plan_is_valid` (dedup):** In the interface-plan deduplication step, disable dedup (return all interfaces). Expected result: test fails because plan length exceeds the bound or contains duplicate hashes.

3. **For `test_large_assembly_plan_is_valid` (operator parsing):** In operator-expression parsing, mishandle range syntax (e.g., parse `"1-60"` as a literal string instead of expanding it). Expected result: test fails because operator IDs don't match the source mmCIF.

4. **For `test_chain_ids_use_label_asym_id_consistently`:** In sequence extraction, use `auth_asym_id` instead of `label_asym_id`. Expected result: test fails because unexpected chain IDs appear in `chain_sequences`.

Document each injection and the observed failure message in a file named `TESTING_VERIFICATION.md`, committed alongside the test suite. This proves the suite has teeth and is not rubber-stamping. The document should have one section per injection with a heading naming the test, a description of the bug injected, and the verbatim pytest failure message observed.

## Step 6 — Confirm Fixture Files Are Committed

Run:

```bash
ls -la tests/fixtures/mmCIF/
ls -la tests/fixtures/expected/
```

All six files (three mmCIFs, three JSONs) must be present and under source control.

## What This Suite Intentionally Does Not Test

Documented non-goals so future contributors don't add tests unnecessarily:

- Sequence-extraction correctness beyond chain-ID convention. Module 2 does its own sequence work and will surface bugs.
- Obsolete-redirection edge cases. Rare; low blast radius when wrong.
- Observed-fraction off-by-one errors. Soft threshold; small errors don't meaningfully change dataset composition.
- Parquet schema drift. Module 2 will fail loudly at load time with a schema mismatch.
- Resolution / R-free / method filters. Mature library code; bugs would be obvious in the Step 20 summary report.
- SHA256 correctness. Well-defined operation using a standard library.

Tests for these areas should be added only when: (1) a real bug is encountered, in which case add the test that would have caught it; (2) interface-planning logic changes substantially, in which case expand Test 2; (3) Module 2 reveals an implicit contract with Module 1 that isn't currently tested, in which case move that assertion upstream.

## Final Implementation Checklist

The agent must complete these steps in order:

1. Download and commit the three fixture mmCIFs (Step 1a).
2. Run the pipeline manually over the fixtures and verify the expected JSON values match reality. Write the three expected JSON files (Step 1b).
3. Implement `conftest.py` (Step 2).
4. Implement `test_module1.py` with all three tests (Step 3).
5. Run the suite; confirm all three tests pass (Step 4).
6. Perform bug-injection verification for each test; document in `TESTING_VERIFICATION.md` (Step 5).
7. Confirm all fixture files are committed (Step 6).
8. Commit everything in a single atomic commit (tests + fixtures + verification doc).