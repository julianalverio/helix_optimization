from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Iterable, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
ASSEMBLY_REST = "https://data.rcsb.org/rest/v1/core/assembly/{entry_id}/{assembly_id}"
INTERFACE_REST = "https://data.rcsb.org/rest/v1/core/interface/{entry_id}/{assembly_id}/{interface_id}"

GRAPHQL_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    exptl { method }
    refine {
      ls_d_res_high
      ls_R_factor_R_free
    }
    em_3d_reconstruction { resolution }
    rcsb_accession_info {
      status_code
      deposit_date
      initial_release_date
    }
    rcsb_entry_info {
      polymer_entity_count
      polymer_entity_count_protein
      polymer_entity_count_DNA
      polymer_entity_count_RNA
      polymer_composition
      resolution_combined
      nonpolymer_entity_count
      deposited_polymer_monomer_count
    }
    assemblies {
      rcsb_id
      pdbx_struct_assembly { id details }
      pdbx_struct_assembly_gen { assembly_id oper_expression asym_id_list }
      rcsb_assembly_info {
        polymer_entity_instance_count
        polymer_entity_instance_count_protein
        polymer_entity_instance_count_DNA
        polymer_entity_instance_count_RNA
      }
    }
    polymer_entities {
      rcsb_id
      entity_poly {
        type
        rcsb_sample_sequence_length
        pdbx_seq_one_letter_code_can
      }
      rcsb_entity_source_organism {
        ncbi_taxonomy_id
        ncbi_scientific_name
      }
    }
  }
}
"""


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "twistr/0.1 (+https://github.com/)"})
    return session


def fetch_all_released_ids(session: requests.Session) -> list[str]:
    body = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.structure_determination_methodology",
                "operator": "exact_match",
                "value": "experimental",
            },
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    response = session.post(SEARCH_URL, json=body, timeout=300)
    response.raise_for_status()
    payload = response.json()
    return [item["identifier"] for item in payload.get("result_set", [])]


def _batches(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _batch_key(ids: list[str]) -> str:
    joined = ",".join(sorted(ids))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def fetch_metadata(
    session: requests.Session,
    ids: Iterable[str],
    cache_dir: Path,
    batch_size: int = 100,
    sleep_between: float = 0.05,
) -> tuple[list[dict], list[str]]:
    ids_list = sorted({i.upper() for i in ids})
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    failed_ids: list[str] = []
    for batch in _batches(ids_list, batch_size):
        cache_file = cache_dir / f"{_batch_key(batch)}.jsonl"
        if cache_file.exists():
            with cache_file.open() as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            continue
        try:
            response = session.post(
                GRAPHQL_URL,
                json={"query": GRAPHQL_QUERY, "variables": {"ids": batch}},
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.warning(
                "GraphQL request failed for batch of %d IDs (%s...): %s",
                len(batch), batch[0], exc,
            )
            failed_ids.extend(batch)
            continue
        if "errors" in payload:
            logger.warning(
                "GraphQL returned errors for batch of %d IDs (%s...): %s",
                len(batch), batch[0], payload["errors"],
            )
            failed_ids.extend(batch)
            continue
        entries = payload.get("data", {}).get("entries") or []
        tmp = cache_file.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        tmp.replace(cache_file)
        results.extend(entries)
        time.sleep(sleep_between)
    return results, failed_ids


def fetch_assembly(session: requests.Session, entry_id: str, assembly_id: str) -> dict:
    url = ASSEMBLY_REST.format(entry_id=entry_id.upper(), assembly_id=assembly_id)
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_interface(session: requests.Session, entry_id: str, assembly_id: str, interface_id: str) -> dict:
    url = INTERFACE_REST.format(
        entry_id=entry_id.upper(), assembly_id=assembly_id, interface_id=interface_id
    )
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.json()
