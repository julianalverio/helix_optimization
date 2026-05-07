from twistr.pipeline.examples.windowing import stable_helix_seed, tile_windows


def test_tile_single_window_when_length_equals_min():
    windows = tile_windows(length=8, seed=0, window_min=8, window_max=15)
    assert windows == [(0, 7)]


def test_tile_single_window_when_length_equals_max():
    windows = tile_windows(length=15, seed=0, window_min=8, window_max=15)
    assert windows == [(0, 14)]


def test_tile_single_window_when_length_between_min_and_max():
    windows = tile_windows(length=11, seed=0, window_min=8, window_max=15)
    assert windows == [(0, 10)]


def test_tile_nothing_when_length_below_min():
    assert tile_windows(length=7, seed=0, window_min=8, window_max=15) == []


def test_tile_multiple_windows_cover_full_length():
    length = 40
    windows = tile_windows(length=length, seed=123, window_min=8, window_max=15)
    assert windows[0][0] == 0
    assert windows[-1][1] == length - 1
    for (s1, e1), (s2, e2) in zip(windows, windows[1:]):
        assert s2 == e1 + 1


def test_tile_deterministic_with_same_seed():
    a = tile_windows(length=50, seed=42, window_min=8, window_max=15)
    b = tile_windows(length=50, seed=42, window_min=8, window_max=15)
    assert a == b


def test_tile_differs_across_seeds():
    a = tile_windows(length=50, seed=1, window_min=8, window_max=15)
    b = tile_windows(length=50, seed=2, window_min=8, window_max=15)
    assert a != b


def test_stable_helix_seed_is_deterministic():
    a = stable_helix_seed(42, "1BRS", 1, 0)
    b = stable_helix_seed(42, "1BRS", 1, 0)
    assert a == b


def test_stable_helix_seed_differs_by_helix_index():
    a = stable_helix_seed(42, "1BRS", 1, 0)
    b = stable_helix_seed(42, "1BRS", 1, 1)
    assert a != b


def test_stable_helix_seed_differs_by_pdb_id():
    a = stable_helix_seed(42, "1ABC", 1, 0)
    b = stable_helix_seed(42, "1XYZ", 1, 0)
    assert a != b
