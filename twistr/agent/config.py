from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for one lead-optimization agent run."""

    # ---- Inputs --------------------------------------------------------
    target_pdb_path: str
    """Multi-chain PDB containing the target plus the two designed helices,
    pre-aligned in a shared frame."""

    helix_chain_ids: tuple[str, ...]
    """Chain IDs of the two designed α-helices."""

    target_chain_ids: tuple[str, ...]
    """Chain IDs of the target / antigen polymer."""

    checkpoint_path: str = "runtime/data/checkpoints/lead_opt_v1/last.ckpt"
    """PyTorch Lightning checkpoint of the trained lead-optimization model.
    Loaded with `ExamplesModule.load_from_checkpoint` — hyperparameters
    travel with the checkpoint, so the runtime cfg is reconstructed from
    the file's `hparams.cfg` payload rather than re-loaded from
    `ml.yaml`."""

    # ---- Claude --------------------------------------------------------
    claude_model: str = "claude-opus-4-7"
    max_tokens_per_turn: int = 4096

    # ---- Search budget -------------------------------------------------
    max_iterations: int = 200
    """Maximum number of model forward passes (mutations evaluated)."""

    initial_mutation_budget: int = 1
    """How many simultaneous mutations the agent is initially permitted
    to propose per turn. Multi-residue jumps are useful late in a run
    once the single-mutation frontier stabilises."""

    # ---- Metric hyperparameters ---------------------------------------
    interface_contact_distance: float = 5.5
    """Heavy-atom distance threshold (Å) for marking interface residues
    and selecting atoms that contribute to the metrics."""

    sc_distance_sigma: float = 1.5
    """Gaussian width (Å) on inter-atom distance for Lawrence-Colman-style
    weighting of the SC metric."""

    ec_distance_cutoff: float = 12.0
    """Pairwise distance cutoff (Å) for the Coulomb sum used to score
    electrostatic complementarity."""

    bsa_probe_radius: float = 1.4
    """SASA probe radius (Å). 1.4 Å is the standard water-probe radius."""

    # ---- Output --------------------------------------------------------
    output_dir: str = "runtime/outputs/agent"
    """Where Pareto-frontier JSONL and per-design transcripts are written."""

    save_intermediate_every: int = 25
    """Flush the Pareto frontier to disk every N iterations so a crash
    doesn't lose a long agent run."""

    # ---- Inference -----------------------------------------------------
    device: str = "cuda"
    """Torch device for the forward pass. Falls back to CPU if CUDA is
    unavailable at startup."""

    # ---- Reproducibility ----------------------------------------------
    seed: int = 0
    """Seeds the agent's tie-breaking RNG (used when Claude proposes an
    ambiguous mutation, e.g. residue index out of range)."""


def load_agent_config(path: Path | str) -> AgentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in fields(AgentConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown agent config keys: {sorted(unknown)}")
    # Coerce tuple-valued fields from list-of-strings YAML literals.
    for tuple_field in ("helix_chain_ids", "target_chain_ids"):
        if tuple_field in raw:
            raw[tuple_field] = tuple(raw[tuple_field])
    return AgentConfig(**raw)
