"""Background ingestion jobs.

Parsing and embedding a PDF takes minutes on a CPU, so ``POST /ingest`` saves the uploads,
queues a job and returns at once; ``GET /ingest/{job_id}`` reports progress. One worker thread
runs the jobs, one at a time, so uploads never compete for the CPU with each other.

Job status is kept in memory: it is lost when the API restarts (the indexed documents are not).
A job is visible only to admins of the tenant that started it.
"""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "failed"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class FileResult:
    source: str
    status: str  # added / updated / unchanged / access_updated / quarantined / failed / queued
    chunks: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class Job:
    job_id: str
    tenant_id: str
    requested_by: str
    status: JobStatus
    files: list[FileResult]
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# (path, source) -> FileResult; runs in the worker thread.
FileRunner = Callable[[Path, str], FileResult]


class IngestJobs:
    def __init__(self, upload_dir: Path, max_jobs_kept: int = 200) -> None:
        self.upload_dir = upload_dir
        self.max_jobs_kept = max_jobs_kept
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")

    def new_upload_dir(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex
        folder = self.upload_dir / job_id
        folder.mkdir(parents=True, exist_ok=True)
        return job_id, folder

    def submit(
        self,
        job_id: str,
        folder: Path,
        files: list[tuple[Path, str]],
        *,
        tenant_id: str,
        requested_by: str,
        run_file: FileRunner,
        on_done: Callable[[Job], None] | None = None,
    ) -> Job:
        job = Job(
            job_id,
            tenant_id,
            requested_by,
            "queued",
            [FileResult(source, "queued") for _, source in files],
        )
        with self._lock:
            self._jobs[job_id] = job
            self._trim()
        self._executor.submit(self._run, job, folder, files, run_file, on_done)
        return job

    def get(self, job_id: str, tenant_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        return job if job is not None and job.tenant_id == tenant_id else None

    def wait_idle(self, timeout: float = 600) -> None:
        """Block until queued jobs have run (tests, shutdown)."""
        self._executor.submit(lambda: None).result(timeout=timeout)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(
        self,
        job: Job,
        folder: Path,
        files: list[tuple[Path, str]],
        run_file: FileRunner,
        on_done: Callable[[Job], None] | None,
    ) -> None:
        job.status, job.started_at = "running", _now()
        try:
            for i, (path, source) in enumerate(files):
                job.files[i] = run_file(path, source)
            job.status = "done"
        except Exception as exc:  # a bug or an outage (e.g. Qdrant down): fail the job
            logger.exception("Ingest job %s failed", job.job_id)
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = _now()
            shutil.rmtree(folder, ignore_errors=True)  # uploads are not kept
            if on_done is not None:
                on_done(job)

    def _trim(self) -> None:
        finished = [j for j in self._jobs.values() if j.status in ("done", "failed")]
        for job in finished[: max(0, len(self._jobs) - self.max_jobs_kept)]:
            self._jobs.pop(job.job_id, None)
