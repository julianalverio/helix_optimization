"""Modal image for the post-MaSIF/ScanNet pipeline stage with REAL
PPI-hotspotID integration.

Each Modal worker handles one PDB. The image bundles:
  - mkdssp + freesasa CLI + AmberTools (tleap, sander, pdb4amber, ambpdb,
    MMPBSA.py) — needed by PPI-hotspotID's critires.sh
  - AutoGluon (TabularPredictor) for the PPI-hotspot model
  - The PPI-hotspotID repo cloned into /ppi
  - A patched critires.sh that skips ConSurf MSA construction (substitutes a
    constant-0.5 conservation column — known scoring degradation; cleanest
    workaround given that real ConSurf needs a 100+ GB sequence DB)
  - The `twistr` package source for the residue-graph + ScanNet + hotspot
    filter logic that runs around the tool calls

The patched critires.sh is base64-injected to avoid shell quoting hazards.
"""
from __future__ import annotations

import base64

import modal


# Patched critires.sh: same flow as the upstream version but with
# environment variables pointing at the in-image tool locations and ConSurf
# replaced by a constant-0.5 conservation column.
_PATCHED_CRITIRES = r"""#!/bin/bash
###################################################################################################
## Patched for the twistr Modal image. Original by Jon Wright (IBMS, Academia Sinica). GLPv3.
## Differences from upstream:
##   - paths point at /opt/conda + /usr/bin instead of /home/programs
##   - ConSurf MSA construction is replaced by a constant grade=5 (mid-scale)
##     conservation column. The AutoGluon model expects 4 features (residue
##     type, conservation, SASA, gas-phase energy); we feed 3 real + 1
##     constant. This is the locked design choice — real ConSurf needs a
##     100+ GB sequence DB, which is out of scope for this pipeline.
###################################################################################################
set -e
export freesasa=/usr/bin/freesasa
export scripts=/ppi
export agpath=$scripts/AutogluonModels/ag-20230915_030535
export dssp=/opt/conda/bin/mkdssp
export AMBERHOME=/opt/conda
export PATH=$AMBERHOME/bin:$PATH
# Prepend conda's libstdc++ so fastai/matplotlib's CXXABI requirement is met
# even on Modal's older host glibc.
export LD_LIBRARY_PATH=/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Sanity check
[ -f input.pdb ] || { echo "input.pdb not found"; exit 1; }

# Check for missing backbone atoms
python3 $scripts/check_bb.py input.pdb missing.txt
if [ -s missing.txt ]; then
    echo "ERROR: missing backbone atoms"
    cat missing.txt
    exit 1
fi
grep '^ATOM  ' input.pdb | tail -1 | cut -c22-22 >| original_chain.txt

# AmberTools energy minimisation + MMPBSA decomposition
$scripts/run_amber.sh

# Per-residue gas-phase energy
python3 $scripts/extract_amber_energies.py

# ConSurf substitute: assemble_data.py wants `<resnum> <grade>` per residue
# (where grade is an integer 1..9). We don't have real conservation, so
# emit grade=5 (middle) for every residue in stability.txt.
awk '{print $2, "5"}' stability.txt > consurf.txt

# SASA percentages
$freesasa --config-file $scripts/protor.config --format=seq post_minix.pdb >| post_mini.sasa
python3 $scripts/sasa_to_perc.py post_mini.sasa | awk '{print $3", "$4", "$8}' >| post_mini.relsasa

# DSSP secondary structure
python3 $scripts/renum_rm_h.py post_minix.pdb post_mini_noh.pdb No
sed -i -e 's/CYX/CYS/' -e 's/CYM/CYS/' -e 's/HID/HIS/' -e 's/HIE/HIS/' -e 's/HIP/HIS/' post_mini_noh.pdb
$dssp post_mini_noh.pdb post_mini_noh.dssp

# Build the per-residue feature table
$scripts/set_numbers.sh
PYTHONPATH=.:$scripts python3 $scripts/assemble_data.py >| assemble.txt

# AutoGluon classifier writes results_ambnum.txt directly via to_csv.
# Don't redirect stdout (scoring.py prints AutoGluon banners which would
# pollute the file). It's OK for this file to be empty if no positives.
PYTHONPATH=.:$scripts python3 $scripts/scoring.py $agpath

# Re-map AMBER → PDB numbering. print_results emits one row per positive;
# `|| true` handles the empty-positives case so grep's exit-1 doesn't kill
# the script under `set -e`.
{ PYTHONPATH=.:$scripts python3 $scripts/print_results.py results_ambnum.txt 2>/dev/null \
    | grep -E -v 'ACE|NME' || true; } >| results.txt

# Convert to `<chain> <resnum> <score>` (1.0 for hotspots; non-hotspots
# are implicit 0.0 by virtue of being absent from results.txt).
awk '{print $3, $2, "1.000"}' results.txt > results_scored.txt
mv results_scored.txt results.txt
"""


def build_image() -> modal.Image:
    """Single Modal image with real PPI-hotspotID + post-processing deps."""
    img = (
        # Python 3.10 + AutoGluon 0.8.2 matches the version the shipped
        # PPI-hotspotID model was serialized with — newer AutoGluon (1.x)
        # renamed internal feature-generator attributes and breaks load.
        modal.Image.micromamba(python_version="3.10")
        .apt_install("freesasa", "git")
        .micromamba_install(
            "ambertools", "dssp",
            channels=["conda-forge", "bioconda"],
        )
        # numpy<2 because AutoGluon 0.8.2 predates the numpy 2.0 ABI break.
        .pip_install(
            "numpy<2", "scipy>=1.11", "gemmi>=0.6.5",
            "pandas>=2.0,<2.2", "pyarrow>=14.0", "pyyaml>=6.0",
            "biopython", "ipython",
            extra_options="--only-binary=:all:",
        )
        # The shipped trainer.pkl stores model paths as lists (a 0.9+ save
        # format) even though metadata.json claims 0.8.2 — so we can't load
        # with 0.8.2. AutoGluon 1.0.0 supports the list path format and
        # still has the `passthrough` attribute on AsTypeFeatureGenerator
        # that 1.5+ renamed to `_fit_passthrough`. fastai is needed because
        # the bagged ensemble has a NeuralNetFastAI child.
        .pip_install(
            "autogluon.tabular[catboost,fastai,lightgbm,xgboost]==1.0.0",
            extra_options="--only-binary=:all:",
        )
        # sklearn 1.4 added monotonic_cst to DecisionTreeClassifier and breaks
        # unpickling 1.3-trained RandomForest models from PPI-hotspotID.
        .pip_install("scikit-learn==1.3.0", extra_options="--only-binary=:all:")
        # Clone PPI-hotspotID and the patched critires.sh.
        .run_commands(
            "rm -rf /ppi && git clone --depth 1 "
            "https://github.com/wrigjz/ppihotspotid /ppi",
        )
        .run_commands(
            f"echo {base64.b64encode(_PATCHED_CRITIRES.encode()).decode()} | "
            "base64 -d > /ppi/critires.sh && chmod +x /ppi/critires.sh",
            # Patch run_amber.sh: replace the upstream /home/programs paths
            # with our in-image equivalents so AmberTools picks up its
            # config files from the conda env and PPI scripts from /ppi.
            "sed -i "
            "-e 's|/home/programs/ambertools20/amber20/amber.sh|/opt/conda/etc/profile.d/conda.sh|g' "
            "-e 's|/home/programs/critires_scripts|/ppi|g' "
            "-e 's|/home/programs/anaconda/linux_202307/init.sh|/opt/conda/etc/profile.d/conda.sh|g' "
            "/ppi/run_amber.sh",
            # Make sander/tleap/etc resolvable via $AMBERHOME at runtime.
            "sed -i 's|^source /opt/conda/etc/profile.d/conda.sh|export AMBERHOME=/opt/conda; export PATH=$AMBERHOME/bin:$PATH|g' /ppi/run_amber.sh",
            # The shipped model's metadata declares AutoGluon 0.8.2; we
            # install 1.0.0 because the trainer.pkl actually uses the
            # list-of-path-components format introduced in 0.9. Bypass the
            # version check so TabularPredictor.load proceeds.
            "sed -i 's|TabularPredictor.load(MODEL_FOLDER_PATH)|TabularPredictor.load(MODEL_FOLDER_PATH, require_version_match=False)|' /ppi/scoring.py",
        )
        .add_local_python_source("twistr")
    )
    return img


pipeline_image = build_image()
