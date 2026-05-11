import numpy as np

from twistr.tensors.constants import (
    ATOM14_SLOT_INDEX,
    ATOM14_SLOT_NAMES,
    DSSP_CHAR_TO_SS8,
    RESIDUE_TYPE_INDEX,
    RESIDUE_TYPE_NAMES,
    SS3_CODES,
    SS8_CODES,
    SS8_TO_SS3,
    atom14_slot_names_array,
    write_constants_npz,
)


def test_residue_ordering():
    assert len(RESIDUE_TYPE_NAMES) == 20
    assert RESIDUE_TYPE_INDEX["ALA"] == 0
    assert RESIDUE_TYPE_INDEX["VAL"] == 19
    assert set(RESIDUE_TYPE_NAMES) == set(RESIDUE_TYPE_INDEX)


def test_atom14_backbone_layout():
    assert len(ATOM14_SLOT_NAMES) == 20
    for slots in ATOM14_SLOT_NAMES:
        assert len(slots) == 14
        assert slots[0] == "N"
        assert slots[1] == "CA"
        assert slots[2] == "C"
        assert slots[3] == "O"


def test_atom14_gly_no_cb():
    assert ATOM14_SLOT_NAMES[RESIDUE_TYPE_INDEX["GLY"]][4] == ""


def test_atom14_trp_fills_fourteen_slots():
    trp_slots = ATOM14_SLOT_NAMES[RESIDUE_TYPE_INDEX["TRP"]]
    assert "" not in trp_slots


def test_atom14_slot_index_consistency():
    for name, slots in zip(RESIDUE_TYPE_NAMES, ATOM14_SLOT_NAMES):
        lookup = ATOM14_SLOT_INDEX[name]
        for atom, slot in lookup.items():
            assert slots[slot] == atom


def test_ss_code_sizes():
    assert len(SS3_CODES) == 4
    assert len(SS8_CODES) == 9


def test_ss_mapping():
    assert DSSP_CHAR_TO_SS8["H"] == 0
    assert DSSP_CHAR_TO_SS8[" "] == 7
    assert SS8_TO_SS3[0] == 0
    assert SS8_TO_SS3[3] == 1
    assert SS8_TO_SS3[7] == 2
    assert SS8_TO_SS3[8] == 3


def test_atom14_array_shape():
    arr = atom14_slot_names_array()
    assert arr.shape == (20, 14)


def test_write_constants_npz(tmp_path):
    out = tmp_path / "constants.npz"
    write_constants_npz(out)
    data = np.load(out)
    assert data["residue_type_names"].shape == (20,)
    assert data["atom14_slot_names"].shape == (20, 14)
    assert data["ss_3_codes"].shape == (4,)
    assert data["ss_8_codes"].shape == (9,)
