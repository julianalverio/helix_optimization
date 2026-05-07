import numpy as np

from twistr.pipeline.examples.constants import SS8_G, SS8_H, SS8_I, SS8_S, SS8_T
from twistr.pipeline.examples.segmentation import (
    filter_by_length,
    find_helix_segments,
    merge_by_gap,
    smooth_ss8,
)


def test_smooth_single_G_between_H_is_rewritten():
    arr = np.array([SS8_H, SS8_H, SS8_G, SS8_H, SS8_H], dtype=np.int8)
    out = smooth_ss8(arr)
    assert out.tolist() == [SS8_H, SS8_H, SS8_H, SS8_H, SS8_H]


def test_smooth_two_T_between_H_is_rewritten():
    arr = np.array([SS8_H, SS8_H, SS8_T, SS8_T, SS8_H, SS8_H], dtype=np.int8)
    out = smooth_ss8(arr)
    assert out.tolist() == [SS8_H, SS8_H, SS8_H, SS8_H, SS8_H, SS8_H]


def test_smooth_three_GIT_between_H_is_not_rewritten():
    arr = np.array([SS8_H, SS8_G, SS8_I, SS8_T, SS8_H], dtype=np.int8)
    out = smooth_ss8(arr)
    assert out.tolist() == [SS8_H, SS8_G, SS8_I, SS8_T, SS8_H]


def test_smooth_not_flanked_by_H_left_untouched():
    arr = np.array([SS8_S, SS8_H, SS8_H], dtype=np.int8)
    out = smooth_ss8(arr)
    assert out.tolist() == [SS8_S, SS8_H, SS8_H]


def test_smooth_does_not_modify_input():
    arr = np.array([SS8_H, SS8_G, SS8_H], dtype=np.int8)
    original = arr.copy()
    smooth_ss8(arr)
    assert arr.tolist() == original.tolist()


def test_find_helix_segments_respects_min_length():
    arr = np.array([SS8_H, SS8_H, SS8_S, SS8_H, SS8_H, SS8_H, SS8_H, SS8_S, SS8_H], dtype=np.int8)
    segments = find_helix_segments(arr, min_length=4)
    assert segments == [(3, 6)]


def test_find_helix_segments_finds_multiple():
    arr = np.array([SS8_H] * 6 + [SS8_S] * 3 + [SS8_H] * 7, dtype=np.int8)
    segments = find_helix_segments(arr, min_length=6)
    assert segments == [(0, 5), (9, 15)]


def test_find_helix_segments_empty_when_too_short():
    arr = np.array([SS8_H, SS8_H, SS8_H], dtype=np.int8)
    assert find_helix_segments(arr, min_length=6) == []


def test_merge_by_gap_merges_within_max_gap():
    contacts = np.zeros(20, dtype=bool)
    contacts[2:5] = True
    contacts[10:13] = True
    merged = merge_by_gap(contacts, 0, 19, max_gap=7)
    assert merged == [(2, 12)]


def test_merge_by_gap_does_not_merge_beyond_max_gap():
    contacts = np.zeros(20, dtype=bool)
    contacts[2:5] = True
    contacts[13:16] = True
    merged = merge_by_gap(contacts, 0, 19, max_gap=7)
    assert merged == [(2, 4), (13, 15)]


def test_merge_by_gap_empty_when_no_contacts():
    contacts = np.zeros(20, dtype=bool)
    merged = merge_by_gap(contacts, 0, 19, max_gap=7)
    assert merged == []


def test_filter_by_length_drops_short_spans():
    spans = [(0, 4), (10, 19), (25, 31)]
    assert filter_by_length(spans, min_length=8) == [(10, 19)]
