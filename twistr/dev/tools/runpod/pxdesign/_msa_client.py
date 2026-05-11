"""Minimal ColabFold MMseqs2 API client. Submits one sequence, polls
until COMPLETE, downloads the tar.gz, extracts uniref + env a3m hits,
and writes a combined `non_pairing.a3m` in the format PXDesign accepts.

Vendored slice of the colabfold-mode branch of Protenix's
`twistr/external/Protenix/protenix/web_service/colab_request_utils.py`
(lines 44-336). We kept only the colabfold path because (a) Protenix's
server returns `0.a3m + .m8` files that need an additional
post-processing pass through Protenix's biotite-heavy stack, and (b)
ColabFold's output `.a3m` format is what PXDesign reads directly.

The pairing MSA is intentionally not produced here — PXDesign's
`Protenix/runner/msa_search.py:182-189` makes `pairedMsaPath` optional;
for single-chain targets (the common case) `unpairedMsaPath` is
sufficient.
"""
from __future__ import annotations

import logging
import tarfile
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

HOST = "https://api.colabfold.com"
HTTP_TIMEOUT_SEC = 6.02         # connect+read timeout for one request
POLL_INTERVAL_SEC = 60          # delay between status polls
RETRY_BACKOFF_SEC = 5           # delay between retries on transient HTTP errors
MAX_ERRORS = 5                  # retry budget per request


class MsaSearchError(RuntimeError):
    pass


def _post(url: str, data: dict) -> dict:
    """POST with retry. Returns parsed JSON or raises MsaSearchError.
    Retries on transient HTTP errors AND non-JSON responses, matching
    the upstream Protenix client."""
    errors = 0
    while True:
        try:
            return requests.post(url, data=data, timeout=HTTP_TIMEOUT_SEC).json()
        except (requests.exceptions.RequestException, ValueError) as e:
            errors += 1
            if errors > MAX_ERRORS:
                raise MsaSearchError(f"POST {url} failed after {errors} retries: {e}")
            logger.warning(f"POST {url} retry {errors}/{MAX_ERRORS}: {e}")
            time.sleep(RETRY_BACKOFF_SEC)


def _get_status(ticket_id: str) -> dict:
    """Poll the ticket. Retries on transient HTTP errors AND non-JSON."""
    errors = 0
    while True:
        try:
            return requests.get(f"{HOST}/ticket/{ticket_id}", timeout=HTTP_TIMEOUT_SEC).json()
        except (requests.exceptions.RequestException, ValueError) as e:
            errors += 1
            if errors > MAX_ERRORS:
                raise MsaSearchError(f"status poll failed after {errors} retries: {e}")
            logger.warning(f"status poll retry {errors}/{MAX_ERRORS}: {e}")
            time.sleep(RETRY_BACKOFF_SEC)


def _download(ticket_id: str, dest: Path) -> None:
    """Pull the result tarball to `dest`."""
    errors = 0
    while True:
        try:
            r = requests.get(f"{HOST}/result/download/{ticket_id}", timeout=HTTP_TIMEOUT_SEC)
            dest.write_bytes(r.content)
            return
        except requests.exceptions.RequestException as e:
            errors += 1
            if errors > MAX_ERRORS:
                raise MsaSearchError(f"download failed after {errors} retries: {e}")
            logger.warning(f"download retry {errors}/{MAX_ERRORS}: {e}")
            time.sleep(RETRY_BACKOFF_SEC)


def _parse_fasta(text: str) -> dict[str, str]:
    """{header → sequence}. Blank/comment lines ignored."""
    result: dict[str, str] = {}
    header: str | None = None
    for line in text.replace("\x00", "").splitlines():
        if line.startswith(">"):
            header = line[1:].strip()
            result[header] = ""
        elif header is not None:
            result[header] += line.strip()
    return result


def fetch_unpaired_msa(sequence: str, work_dir: Path) -> Path:
    """Submit `sequence` to ColabFold's unpaired-MSA endpoint, wait for
    completion, write `<work_dir>/non_pairing.a3m`, return its path.

    Wall clock: typically 5-30 min depending on server queue + sequence
    length. Caller is responsible for parallelizing across chains via
    threads (this function blocks)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    submit_url = f"{HOST}/ticket/msa"
    # ColabFold's MMseqs2 server requires FASTA (with `>` header), not a bare
    # sequence. Bare submission gets accepted as a ticket but fails server-side
    # with status=ERROR.
    submit_data = {"q": f">query\n{sequence}\n", "mode": "env"}  # env = filtered + envdb

    submission = _post(submit_url, submit_data)
    while submission.get("status") in ("UNKNOWN", "RATELIMIT"):
        logger.info(f"server busy ({submission['status']}); resubmitting in 60s")
        time.sleep(POLL_INTERVAL_SEC)
        submission = _post(submit_url, submit_data)
    if submission.get("status") in ("ERROR", "MAINTENANCE"):
        raise MsaSearchError(f"server returned {submission['status']}: check sequence validity")

    ticket_id = submission["id"]
    state = submission
    while state.get("status") in ("UNKNOWN", "PENDING", "RUNNING"):
        time.sleep(POLL_INTERVAL_SEC)
        state = _get_status(ticket_id)
    if state.get("status") != "COMPLETE":
        raise MsaSearchError(f"ticket {ticket_id} ended in status {state.get('status')}")

    tar_path = work_dir / "out.tar.gz"
    _download(ticket_id, tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(work_dir)
    tar_path.unlink()

    # ColabFold unpaired output: uniref.a3m + bfd.mgnify30.metaeuk30.smag30.a3m.
    # Concatenate (skipping the server's query echoes) into a single
    # non_pairing.a3m, matching Protenix's colabfold post-process at
    # colab_request_utils.py:292-311.
    env = _parse_fasta((work_dir / "bfd.mgnify30.metaeuk30.smag30.a3m").read_text())
    uniref = _parse_fasta((work_dir / "uniref.a3m").read_text())

    out_path = work_dir / "non_pairing.a3m"
    with out_path.open("w") as f:
        f.write(f">query\n{sequence}\n")
        for hits in (env, uniref):
            for header, seq in hits.items():
                if not header.startswith("query_"):
                    f.write(f">{header}\n{seq}\n")
    return out_path
