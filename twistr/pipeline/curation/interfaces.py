from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from . import rcsb

logger = logging.getLogger(__name__)


class InterfaceFetchError(Exception):
    pass


@dataclass
class InterfacePlanEntry:
    entity_id_1: str | None
    entity_id_2: str | None
    asym_id_1: str | None
    asym_id_2: str | None
    interface_id: str
    interface_cluster_id: str | None
    interface_area: float | None


def _extract_partner_info(interface: dict) -> list[tuple[str | None, str | None]]:
    partners = interface.get("rcsb_interface_partner") or []
    out: list[tuple[str | None, str | None]] = []
    for p in partners:
        ident = p.get("interface_partner_identifier") or {}
        out.append((ident.get("entity_id"), ident.get("asym_id")))
    return out


def _extract_area(interface: dict) -> float | None:
    info = interface.get("rcsb_interface_info") or {}
    val = info.get("interface_area")
    return float(val) if val is not None else None


def _extract_num_residues(interface: dict) -> int | None:
    info = interface.get("rcsb_interface_info") or {}
    val = info.get("num_interface_residues")
    return int(val) if val is not None else None


def _dedupe_key(
    entity_pair: tuple[str | None, str | None], area: float | None, n_residues: int | None
) -> tuple:
    sorted_pair = tuple(sorted((entity_pair[0] or "", entity_pair[1] or "")))
    area_bucket = int(area / 50) if area is not None else None
    residue_bucket = int(n_residues / 3) if n_residues is not None else None
    return (sorted_pair, area_bucket, residue_bucket)


def dedupe_from_responses(
    assembly: dict, interfaces_by_id: dict[str, dict]
) -> list[InterfacePlanEntry]:
    ids_container = assembly.get("rcsb_assembly_container_identifiers") or {}
    interface_ids = ids_container.get("interface_ids") or []
    seen: set[tuple] = set()
    plan: list[InterfacePlanEntry] = []
    for interface_id in interface_ids:
        interface = interfaces_by_id.get(str(interface_id))
        if interface is None:
            continue
        partners = _extract_partner_info(interface)
        p1 = partners[0] if len(partners) > 0 else (None, None)
        p2 = partners[1] if len(partners) > 1 else (None, None)
        area = _extract_area(interface)
        n_res = _extract_num_residues(interface)
        key = _dedupe_key((p1[0], p2[0]), area, n_res)
        if key in seen:
            continue
        seen.add(key)
        plan.append(
            InterfacePlanEntry(
                entity_id_1=p1[0],
                entity_id_2=p2[0],
                asym_id_1=p1[1],
                asym_id_2=p2[1],
                interface_id=str(interface_id),
                interface_cluster_id=None,
                interface_area=area,
            )
        )
    return plan


def fetch_unique_interfaces(
    session: requests.Session, entry_id: str, assembly_id: str
) -> list[InterfacePlanEntry]:
    try:
        assembly = rcsb.fetch_assembly(session, entry_id, assembly_id)
    except requests.RequestException as exc:
        logger.warning("assembly fetch failed for %s/%s: %s", entry_id, assembly_id, exc)
        raise InterfaceFetchError(f"assembly fetch failed for {entry_id}/{assembly_id}: {exc}") from exc

    ids_container = assembly.get("rcsb_assembly_container_identifiers") or {}
    interface_ids = ids_container.get("interface_ids") or []
    interfaces_by_id: dict[str, dict] = {}
    for interface_id in interface_ids:
        try:
            interfaces_by_id[str(interface_id)] = rcsb.fetch_interface(
                session, entry_id, assembly_id, str(interface_id)
            )
        except requests.RequestException as exc:
            logger.warning(
                "interface fetch failed for %s/%s/%s: %s",
                entry_id, assembly_id, interface_id, exc,
            )
    return dedupe_from_responses(assembly, interfaces_by_id)


def plan_to_records(plan: list[InterfacePlanEntry]) -> list[dict]:
    return [p.__dict__ for p in plan]
