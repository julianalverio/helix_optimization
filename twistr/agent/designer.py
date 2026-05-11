"""Loads a multi-chain target+helices PDB and a trained lead-optimization
checkpoint, exposes per-residue mutation + forward-pass + scoring API for
the agent loop.

Convention: both designed helices share `chain_slot = 0` (the "design"
group from the model's perspective); target chains are concatenated into
`chain_slot = 1`. `is_helix` is True for every helix residue regardless
of which chain it came from. The model was trained with one-helix-per-
example; running it on two-helix complexes co-opts the architecture's
single-design-group convention and the model treats the helix-helix
pair as one extended design region.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
import scipy.spatial
import torch

from twistr.pipeline.constants import COORD_SCALE_ANGSTROMS
from twistr.pipeline.features.builder import build_features
from twistr.pipeline.models.lightning_module import ExamplesModule
from twistr.pipeline.models.sidechain import apply_torsions_to_atom14
from twistr.tensors.constants import (
    ATOM14_SLOT_INDEX,
    ATOM14_SLOT_NAMES,
    RESIDUE_TYPE_INDEX,
    RESIDUE_TYPE_NAMES,
)

ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
THREE_LETTER = {v: k for k, v in ONE_LETTER.items()}


@dataclass
class ChainSpan:
    """Where a single input chain sits inside the concatenated batch tensor."""
    chain_id: str
    is_helix: bool
    start: int                # inclusive index into the (N,)-axis arrays
    end: int                  # exclusive
    residue_numbers: list[int]  # 1-indexed author-numbered residue IDs from the PDB


@dataclass
class Prediction:
    """Output of one model forward pass on the current state."""
    atoms_atom14_ang: torch.Tensor       # (1, N, 14, 3) — predicted heavy atoms in Å
    interaction_matrix: torch.Tensor     # (1, N, N, 6) — sigmoid(IM logits)
    residue_type: torch.Tensor           # (1, N) long
    chain_slot: torch.Tensor             # (1, N) long
    is_helix: torch.Tensor               # (1, N) bool
    is_interface_residue: torch.Tensor   # (1, N) bool
    atom_mask: torch.Tensor              # (1, N, 14) int8 — 1 = present


class Designer:
    """Loads the target+helices structure and the lead-optimization model,
    and exposes apply_mutation / revert / predict / helix_sequence for the
    agent loop."""

    def __init__(
        self,
        target_pdb: Path,
        helix_chain_ids: tuple[str, ...],
        target_chain_ids: tuple[str, ...],
        checkpoint_path: Path,
        device: str = "cuda",
        interface_contact_distance: float = 5.5,
    ):
        self.target_pdb = Path(target_pdb)
        self.helix_chain_ids = tuple(helix_chain_ids)
        self.target_chain_ids = tuple(target_chain_ids)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.interface_contact_distance = interface_contact_distance

        self.module = self._load_module(Path(checkpoint_path))
        self.batch, self.chain_spans = self._parse_pdb(self.target_pdb)
        self._initial_residue_type = self.batch["residue_type"].clone()
        self._spans_by_chain = {span.chain_id: span for span in self.chain_spans}

    # ------------------------------------------------------------------
    # Loading

    def _load_module(self, checkpoint_path: Path) -> ExamplesModule:
        module = ExamplesModule.load_from_checkpoint(
            str(checkpoint_path),
            map_location=self.device,
            strict=True,
        )
        module.eval()
        module.freeze()
        return module.to(self.device)

    def _parse_pdb(
        self, pdb_path: Path,
    ) -> tuple[dict[str, torch.Tensor], list[ChainSpan]]:
        """Parse a multi-chain PDB into the batch dict the model consumes.

        Output keys mirror `pad_collate`'s schema in `pipeline/datasets/
        datamodule.py`: coordinates (centered + /10), atom_mask, residue_type,
        chain_slot, is_helix, is_interface_residue, padding_mask. All
        tensors are on `self.device` with a leading batch dimension of 1.
        """
        st = gemmi.read_structure(str(pdb_path))
        st.setup_entities()
        model = st[0]

        chain_order = list(self.helix_chain_ids) + list(self.target_chain_ids)
        per_chain: list[ChainSpan] = []

        coord_blocks: list[np.ndarray] = []
        mask_blocks: list[np.ndarray] = []
        rtype_blocks: list[np.ndarray] = []
        chain_slot_blocks: list[np.ndarray] = []
        is_helix_blocks: list[np.ndarray] = []

        cursor = 0
        for chain_id in chain_order:
            chain = model.find_chain(chain_id)
            if chain is None:
                raise ValueError(f"chain {chain_id!r} not found in {pdb_path.name}")
            is_helix_chain = chain_id in self.helix_chain_ids
            slot = 0 if is_helix_chain else 1

            chain_coords, chain_mask, chain_rtype, chain_resnums = [], [], [], []
            for residue in chain:
                if residue.name not in RESIDUE_TYPE_INDEX:
                    continue
                slot_idx = ATOM14_SLOT_INDEX[residue.name]
                coords = np.zeros((14, 3), dtype=np.float32)
                mask = np.zeros(14, dtype=np.int8)
                for atom in residue:
                    if atom.name not in slot_idx:
                        continue
                    i = slot_idx[atom.name]
                    coords[i] = (atom.pos.x, atom.pos.y, atom.pos.z)
                    mask[i] = 1
                if mask[1] == 0:                                  # require CA
                    continue
                chain_coords.append(coords)
                chain_mask.append(mask)
                chain_rtype.append(RESIDUE_TYPE_INDEX[residue.name])
                chain_resnums.append(residue.seqid.num)

            n = len(chain_coords)
            if n == 0:
                raise ValueError(f"chain {chain_id!r} contains no usable residues")

            per_chain.append(ChainSpan(
                chain_id=chain_id, is_helix=is_helix_chain,
                start=cursor, end=cursor + n, residue_numbers=chain_resnums,
            ))
            cursor += n

            coord_blocks.append(np.stack(chain_coords))
            mask_blocks.append(np.stack(chain_mask))
            rtype_blocks.append(np.array(chain_rtype, dtype=np.int64))
            chain_slot_blocks.append(np.full(n, slot, dtype=np.int64))
            is_helix_blocks.append(np.full(n, is_helix_chain, dtype=bool))

        coords = np.concatenate(coord_blocks, axis=0)             # (N, 14, 3)
        atom_mask = np.concatenate(mask_blocks, axis=0)
        residue_type = np.concatenate(rtype_blocks, axis=0)
        chain_slot = np.concatenate(chain_slot_blocks, axis=0)
        is_helix = np.concatenate(is_helix_blocks, axis=0)

        # Center on the centroid of present heavy atoms; scale to nm.
        present = (atom_mask == 1)
        centroid = coords[present].mean(axis=0)
        coords = (coords - centroid) / COORD_SCALE_ANGSTROMS

        # Interface residues: any helix residue within `interface_contact_distance`
        # of a target heavy atom, or vice versa. Symmetric and cross-chain only.
        is_interface = self._compute_interface_residues(
            coords * COORD_SCALE_ANGSTROMS, atom_mask, is_helix,
        )

        device = self.device
        batch = {
            "coordinates": torch.from_numpy(coords).unsqueeze(0).to(device),
            "atom_mask": torch.from_numpy(atom_mask).unsqueeze(0).to(device),
            "residue_type": torch.from_numpy(residue_type).unsqueeze(0).to(device),
            "chain_slot": torch.from_numpy(chain_slot).unsqueeze(0).to(device),
            "is_helix": torch.from_numpy(is_helix).unsqueeze(0).to(device),
            "is_interface_residue": torch.from_numpy(is_interface).unsqueeze(0).to(device),
            "padding_mask": torch.ones(1, coords.shape[0], dtype=torch.bool, device=device),
        }
        return batch, per_chain

    def _compute_interface_residues(
        self, coords_ang: np.ndarray, atom_mask: np.ndarray, is_helix: np.ndarray,
    ) -> np.ndarray:
        """Mark residues with at least one heavy atom within
        `interface_contact_distance` of a residue on the opposite side
        of the helix/target split."""
        helix_atoms = coords_ang[is_helix][atom_mask[is_helix] == 1]
        target_atoms = coords_ang[~is_helix][atom_mask[~is_helix] == 1]
        if helix_atoms.size == 0 or target_atoms.size == 0:
            return np.zeros(len(is_helix), dtype=bool)

        helix_tree = scipy.spatial.cKDTree(helix_atoms)
        target_tree = scipy.spatial.cKDTree(target_atoms)

        is_interface = np.zeros(len(is_helix), dtype=bool)
        # For each residue, check if any of its present atoms lies within
        # `interface_contact_distance` of an atom on the opposite side.
        for i, side_is_helix in enumerate(is_helix):
            present = atom_mask[i] == 1
            if not present.any():
                continue
            xyz = coords_ang[i, present]
            opposite = target_tree if side_is_helix else helix_tree
            d, _ = opposite.query(xyz, k=1, distance_upper_bound=self.interface_contact_distance)
            if np.any(d < self.interface_contact_distance):
                is_interface[i] = True
        return is_interface

    # ------------------------------------------------------------------
    # Mutation API

    def apply_mutation(self, chain_id: str, position: int, new_residue: str) -> int:
        """Apply a point mutation. `position` is the 1-indexed PDB residue
        number; `new_residue` is the three-letter code. Returns the
        previous residue-type integer index, which can be passed to
        `revert_mutation` for rollback.

        The atom_mask row at this position is rebuilt from scratch to
        match the new residue's atom-14 slot layout — both *enabling*
        slots that the new type owns and *zeroing* slots it doesn't. The
        coordinate tensor at newly-enabled slots starts at zero;
        `apply_torsions_to_atom14` in `predict()` resynthesises those
        positions from the predicted frame + χ angles, so the zero
        initialisation never reaches a metric."""
        n_idx = self._chain_pos_to_n(chain_id, position)
        new_residue = new_residue.upper()
        if new_residue not in RESIDUE_TYPE_INDEX:
            raise ValueError(f"unknown residue {new_residue!r}")
        new_idx = RESIDUE_TYPE_INDEX[new_residue]
        prev = int(self.batch["residue_type"][0, n_idx].item())
        self.batch["residue_type"][0, n_idx] = new_idx
        names = ATOM14_SLOT_NAMES[new_idx]
        row = self.batch["atom_mask"][0, n_idx]
        for slot in range(14):
            row[slot] = 1 if names[slot] else 0
        return prev

    def revert_mutation(self, chain_id: str, position: int, previous_residue_idx: int) -> None:
        """Restore the residue type and rebuild its atom_mask row to
        match. Equivalent to `apply_mutation(chain_id, position, prev_3letter)`
        — the two paths share the same mask-rebuilding logic."""
        self.apply_mutation(
            chain_id, position, RESIDUE_TYPE_NAMES[previous_residue_idx],
        )

    def helix_sequences(self) -> dict[str, str]:
        """Current 1-letter sequences of the designed helices, keyed by
        chain ID."""
        out: dict[str, str] = {}
        rtype = self.batch["residue_type"][0]
        for span in self.chain_spans:
            if not span.is_helix:
                continue
            three = [RESIDUE_TYPE_NAMES[int(rtype[i])] for i in range(span.start, span.end)]
            out[span.chain_id] = "".join(ONE_LETTER[r] for r in three)
        return out

    def residue_at(self, chain_id: str, position: int) -> str:
        """Three-letter code of the residue at (chain, 1-indexed position)."""
        n_idx = self._chain_pos_to_n(chain_id, position)
        return RESIDUE_TYPE_NAMES[int(self.batch["residue_type"][0, n_idx].item())]

    # ------------------------------------------------------------------
    # Inference

    @torch.no_grad()
    def predict(self) -> Prediction:
        """Run the model forward pass on the current state. Returns
        predicted atom-14 coordinates in Å, sigmoid-applied interaction
        matrix probabilities, and per-residue metadata aligned with the
        N axis."""
        features = build_features(self.batch, self.module.cfg)
        out = self.module.model(features)
        atoms_atom14 = apply_torsions_to_atom14(
            out["rotation"], out["translation"], out["torsion_sincos"],
            self.batch["residue_type"],
        )
        atoms_ang = atoms_atom14 * COORD_SCALE_ANGSTROMS
        im_probs = torch.sigmoid(out["interaction_matrix"])
        return Prediction(
            atoms_atom14_ang=atoms_ang,
            interaction_matrix=im_probs,
            residue_type=self.batch["residue_type"],
            chain_slot=self.batch["chain_slot"],
            is_helix=self.batch["is_helix"],
            is_interface_residue=self.batch["is_interface_residue"],
            atom_mask=self.batch["atom_mask"],
        )

    # ------------------------------------------------------------------
    # Output

    def write_pdb(self, path: Path, prediction: Prediction | None = None) -> None:
        """Write the current prediction (or a supplied one) to a PDB file.
        Atom records carry the predicted heavy-atom coordinates in Å with
        the original input chain IDs and PDB residue numbers — so a written
        Pareto design can be opened in PyMOL alongside the wild-type input
        for direct comparison."""
        if prediction is None:
            prediction = self.predict()

        coords = prediction.atoms_atom14_ang[0].detach().cpu().numpy()
        atom_mask = prediction.atom_mask[0].detach().cpu().numpy()
        residue_type = prediction.residue_type[0].detach().cpu().numpy()

        st = gemmi.Structure()
        st.spacegroup_hm = "P 1"
        st.name = path.stem
        model = gemmi.Model("1")

        for span in self.chain_spans:
            chain = gemmi.Chain(span.chain_id)
            for offset, residue_number in enumerate(span.residue_numbers):
                n_idx = span.start + offset
                r_idx = int(residue_type[n_idx])
                res_name = RESIDUE_TYPE_NAMES[r_idx]
                names = ATOM14_SLOT_NAMES[r_idx]
                residue = gemmi.Residue()
                residue.name = res_name
                residue.seqid = gemmi.SeqId(residue_number, " ")
                for slot in range(14):
                    if atom_mask[n_idx, slot] != 1 or not names[slot]:
                        continue
                    atom = gemmi.Atom()
                    atom.name = names[slot]
                    atom.element = gemmi.Element(names[slot][0])
                    x, y, z = coords[n_idx, slot]
                    atom.pos = gemmi.Position(float(x), float(y), float(z))
                    atom.occ = 1.0
                    atom.b_iso = 0.0
                    residue.add_atom(atom)
                if len(residue) > 0:
                    chain.add_residue(residue)
            model.add_chain(chain)
        st.add_model(model)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        st.write_pdb(str(path))

    # ------------------------------------------------------------------
    # Internals

    def _chain_pos_to_n(self, chain_id: str, position: int) -> int:
        if chain_id not in self._spans_by_chain:
            raise ValueError(f"unknown chain {chain_id!r}; expected one of {list(self._spans_by_chain)}")
        span = self._spans_by_chain[chain_id]
        try:
            offset = span.residue_numbers.index(position)
        except ValueError:
            raise ValueError(
                f"residue {position} not present on chain {chain_id!r} "
                f"(available: {span.residue_numbers[0]}..{span.residue_numbers[-1]})"
            ) from None
        return span.start + offset
