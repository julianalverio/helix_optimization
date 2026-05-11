import shutil
import subprocess

import pytest

from twistr.curation import paths
from twistr.curation.config import Config
from twistr.curation import verify

pytestmark = pytest.mark.network


def _rsync_single(pdb_id: str, dest_root):
    pdb_id = pdb_id.lower()
    mid = pdb_id[1:3]
    target_dir = dest_root / "pdb" / mid
    target_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-rlpt",
            "-z",
            "--port=33444",
            f"rsync.rcsb.org::ftp_data/structures/divided/mmCIF/{mid}/{pdb_id}.cif.gz",
            str(target_dir) + "/",
        ],
        check=True,
    )


@pytest.fixture(scope="module")
def rsynced_root(tmp_path_factory):
    if shutil.which("rsync") is None:
        pytest.skip("rsync not available")
    root = tmp_path_factory.mktemp("rsynced")
    for pid in ("1BRS", "1A3N"):
        _rsync_single(pid, root)
    return root


def test_parse_1brs(rsynced_root):
    result = verify.parse_structure(paths.mmcif_abs_path(rsynced_root, "1BRS"))
    assert result.parse_ok
    assert result.method == "X-RAY DIFFRACTION"
    assert result.n_protein_chains >= 2
    assert result.max_protein_observed_fraction > 0.5


def test_parse_1a3n_hemoglobin(rsynced_root):
    result = verify.parse_structure(paths.mmcif_abs_path(rsynced_root, "1A3N"))
    assert result.parse_ok
    assert result.n_protein_chains == 4
    assert result.max_protein_observed_fraction > 0.9


def test_parse_corrupt_file_does_not_raise(tmp_path):
    import gzip
    bad_dir = tmp_path / "pdb" / "xx"
    bad_dir.mkdir(parents=True)
    bad_file = bad_dir / "xxxx.cif.gz"
    with gzip.open(bad_file, "wb") as f:
        f.write(b"not a real mmcif file\n")
    result = verify.parse_structure(bad_file)
    assert not result.parse_ok
    assert result.parse_error is not None
    assert result.pdb_id == "XXXX"
