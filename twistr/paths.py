from __future__ import annotations

from pathlib import Path


def data_root(data_root: str) -> Path:
    root = Path(data_root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / data_root
    return root


def mmcif_rel_path(pdb_id: str) -> str:
    pdb_id = pdb_id.lower()
    return f"pdb/{pdb_id[1:3]}/{pdb_id}.cif.gz"


def mmcif_abs_path(data_root_path: Path, pdb_id: str) -> Path:
    return data_root_path / mmcif_rel_path(pdb_id)


def tensor_rel_path(pdb_id: str, assembly_id: int | str) -> str:
    pdb_id = pdb_id.lower()
    return f"tensors/{pdb_id[1:3]}/{pdb_id}_{assembly_id}.npz"


def tensor_abs_path(output_dir: Path, pdb_id: str, assembly_id: int | str) -> Path:
    return output_dir / tensor_rel_path(pdb_id, assembly_id)


def example_rel_path(pdb_id: str, assembly_id: int | str, example_id: int | str) -> str:
    pdb_id = pdb_id.lower()
    return f"examples/{pdb_id[1:3]}/{pdb_id}_{assembly_id}_{example_id}.npz"


def example_abs_path(output_dir: Path, pdb_id: str, assembly_id: int | str, example_id: int | str) -> Path:
    return output_dir / example_rel_path(pdb_id, assembly_id, example_id)


def marker_rel_path(pdb_id: str, assembly_id: int | str) -> str:
    pdb_id = pdb_id.lower()
    return f"markers/{pdb_id[1:3]}/{pdb_id}_{assembly_id}.marker"


def marker_abs_path(output_dir: Path, pdb_id: str, assembly_id: int | str) -> Path:
    return output_dir / marker_rel_path(pdb_id, assembly_id)


def manifests_dir(data_root_path: Path) -> Path:
    return data_root_path / "manifests"


def aux_dir(data_root_path: Path) -> Path:
    return data_root_path / "aux"


def pdb_dir(data_root_path: Path) -> Path:
    return data_root_path / "pdb"


def cache_dir(data_root_path: Path, config_hash: str, phase: str) -> Path:
    return manifests_dir(data_root_path) / ".cache" / config_hash / phase


def ensure_dirs(data_root_path: Path) -> None:
    for d in (manifests_dir(data_root_path), aux_dir(data_root_path), pdb_dir(data_root_path)):
        d.mkdir(parents=True, exist_ok=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
