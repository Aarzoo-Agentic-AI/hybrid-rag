"""The API routes. Every route except ``/health`` needs ``X-API-Key``; the caller's principal
scopes everything it reads or changes."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Security,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from langsmith import tracing_context

from hybrid_rag.access import AccessPolicy
from hybrid_rag.api.auth import Caller
from hybrid_rag.api.feedback import RecentRequest, run_id_for
from hybrid_rag.api.jobs import FileResult, Job
from hybrid_rag.api.logs import request_id_var, tenant_var
from hybrid_rag.api.schemas import (
    AccessOut,
    DocumentOut,
    ErrorOut,
    FeedbackRequest,
    FeedbackResponse,
    JobOut,
    QueryRequest,
    QueryResponse,
)
from hybrid_rag.api.services import Services
from hybrid_rag.generation import Answer
from hybrid_rag.indexing import ingest_file
from hybrid_rag.retrieval import RelevanceFilters

logger = logging.getLogger(__name__)
router = APIRouter()

_api_key = APIKeyHeader(name="X-API-Key", auto_error=False, description="Your API key")
_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorOut, "description": "Missing or unknown API key"},
    403: {"model": ErrorOut, "description": "This route needs a tenant-admin key"},
    404: {"model": ErrorOut, "description": "Not found, or not visible to you"},
    503: {"model": ErrorOut, "description": "Qdrant or every LLM provider is unavailable"},
}


def services(request: Request) -> Services:
    return request.app.state.services


def caller(
    request: Request,
    key: Annotated[str | None, Security(_api_key)],
    svc: Annotated[Services, Depends(services)],
) -> Caller:
    found = svc.keys.authenticate(key)
    if found is None:
        raise HTTPException(401, "Missing or unknown API key (send it as X-API-Key)")
    request.state.caller = found
    tenant_var.set(found.principal.tenant_id)
    return found


def admin(who: Annotated[Caller, Depends(caller)]) -> Caller:
    if not who.principal.admin:
        raise HTTPException(403, "This route needs a tenant-admin API key")
    return who


def _trace_context(svc: Services, who: Caller, route: str, **metadata: Any):
    """LangSmith metadata and tags for this request's trace (a no-op when tracing is off)."""
    if svc.trace_client is None:
        return nullcontext()
    return tracing_context(
        client=svc.trace_client,
        metadata={
            "request_id": request_id_var.get(),
            "tenant_id": who.principal.tenant_id,
            "route": route,
            **metadata,
        },
        tags=[f"route:{route}", f"tenant:{who.principal.tenant_id}"],
    )


def _audit_base(who: Caller, route: str) -> dict[str, Any]:
    p = who.principal
    return {
        "request_id": request_id_var.get(),
        "route": route,
        "tenant_id": p.tenant_id,
        "user_id": p.user_id,
        "key": who.key_name,
    }


# ------------------------------------------------------------------ query


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={**_ERRORS, 200: {"description": "The answer (or a text/event-stream if stream)"}},
    tags=["query"],
)
def query(
    body: QueryRequest,
    who: Annotated[Caller, Depends(caller)],
    svc: Annotated[Services, Depends(services)],
) -> Any:
    """Ask a question. The answer uses only the policies you may read, with citations.

    With `stream: true` the response is server-sent events: `token` events while the answer is
    written, then one `answer` event with the same JSON as the normal response.
    """
    filters = RelevanceFilters(**body.filters.model_dump()) if body.filters else None
    history = [("human" if t.role == "user" else "ai", t.content) for t in body.history]
    mode = body.mode or svc.settings.retrieval.mode
    request_id = request_id_var.get() or ""
    started = time.perf_counter()
    kwargs = {
        "history": history,
        "mode": body.mode,
        "filters": filters,
        "strategy": body.strategy,
        "run_id": run_id_for(request_id) if request_id else None,  # for feedback on the trace
    }

    def audit(answer: Answer, status_code: int = 200) -> None:
        svc.audit.write(
            {
                **_audit_base(who, "/query"),
                "status_code": status_code,
                "mode": mode,
                "strategy": answer.strategy,
                "stream": body.stream,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                **svc.audit.answer_fields(answer),
            }
        )
        p = who.principal
        svc.recent.remember(
            RecentRequest(
                request_id, p.tenant_id, p.user_id, answer.question, answer.status, time.time()
            )
        )
        if svc.runtime_eval is not None:
            svc.runtime_eval.maybe_check(
                request_id=request_id, tenant_id=p.tenant_id, user_id=p.user_id, answer=answer
            )

    if not body.stream:
        with _trace_context(svc, who, "/query", mode=mode):
            answer = svc.answers.answer(body.question, who.principal, **kwargs)
        audit(answer)
        return QueryResponse.from_answer(answer, request_id, debug=body.debug)

    def events() -> Iterator[str]:
        request_id_var.set(request_id)  # the generator may run in another context
        with _trace_context(svc, who, "/query", mode=mode, stream=True):
            for event in svc.answers.stream(body.question, who.principal, **kwargs):
                if event["type"] == "token":
                    yield _sse("token", {"text": event["text"]})
                else:
                    answer = event["answer"]
                    audit(answer)
                    out = QueryResponse.from_answer(answer, request_id, debug=body.debug)
                    yield _sse("answer", out.model_dump())

    return StreamingResponse(events(), media_type="text/event-stream")


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={**_ERRORS, 409: {"model": ErrorOut, "description": "Already rated"}},
    tags=["query"],
)
def feedback(
    body: FeedbackRequest,
    background: BackgroundTasks,
    who: Annotated[Caller, Depends(caller)],
    svc: Annotated[Services, Depends(services)],
) -> FeedbackResponse:
    """Rate an answer you received: `up` or `down`, with an optional comment.

    Only the caller who asked can rate, within `api.feedback_window_hours`, once per answer.
    The comment is masked like questions. **A thumbs-down also stores your (masked) question**,
    so it can be added to the evaluation set.
    """
    p = who.principal
    asked = svc.recent.lookup(body.request_id, p.tenant_id, p.user_id)
    if asked is None:
        raise HTTPException(404, "No such recent request of yours")
    if not svc.recent.mark_rated(body.request_id):
        raise HTTPException(409, "This answer has already been rated")
    comment = svc.sanitizer.redact_text(body.comment)[0] if body.comment else None
    record = {
        "request_id": body.request_id,
        "tenant_id": p.tenant_id,
        "user_id": p.user_id,
        "key": who.key_name,
        "rating": body.rating,
        "comment": comment,
        "answer_status": asked.answer_status,
    }
    if body.rating == "down":
        record["question"] = asked.question  # already masked; for the golden set
    svc.feedback.write(record)
    svc.audit.write(
        {
            **_audit_base(who, "/feedback"),
            "rated_request_id": body.request_id,
            "rating": body.rating,
            "has_comment": comment is not None,
        }
    )
    if svc.langsmith is not None:
        score = 1 if body.rating == "up" else 0
        background.add_task(svc.langsmith.send, body.request_id, "user_rating", score, comment)
    return FeedbackResponse(request_id=body.request_id, rating=body.rating)


# -------------------------------------------------------------- documents


@router.get("/documents", response_model=list[DocumentOut], responses=_ERRORS, tags=["documents"])
def list_documents(
    who: Annotated[Caller, Depends(caller)],
    svc: Annotated[Services, Depends(services)],
) -> list[DocumentOut]:
    """The documents you may read. Admins see every document of their tenant, with its grants."""
    docs = svc.index.list_documents(who.principal)
    out = []
    for d in docs:
        grants = (
            {
                "allowed_groups": d.allowed_groups,
                "allowed_roles": d.allowed_roles,
                "allowed_users": d.allowed_users,
            }
            if who.principal.admin
            else {}
        )
        out.append(
            DocumentOut(
                doc_id=d.doc_id,
                source=d.source,
                title=d.title,
                chunk_count=d.chunk_count,
                classification=d.classification,
                visibility=d.visibility,
                access_version=d.access_version,
                ingested_at=d.ingested_at,
                pii_status=d.pii_status,
                **grants,
            )
        )
    return out


@router.delete("/documents/{doc_id}", responses=_ERRORS, tags=["documents"])
def delete_document(
    doc_id: str,
    who: Annotated[Caller, Depends(admin)],
    svc: Annotated[Services, Depends(services)],
) -> dict[str, Any]:
    """Delete a document and all its chunks (tenant admins only)."""
    info = svc.index.get_document(doc_id, who.principal)
    if info is None:
        raise HTTPException(404, "No such document")
    removed = svc.index.delete(doc_id, who.principal)
    svc.overrides.set(who.principal.tenant_id, info.source, None)
    svc.audit.write(
        {
            **_audit_base(who, "DELETE /documents"),
            "doc_id": doc_id,
            "source": info.source,
            "chunks_removed": removed,
        }
    )
    return {"doc_id": doc_id, "source": info.source, "chunks_removed": removed}


@router.put(
    "/documents/{doc_id}/access", response_model=AccessOut, responses=_ERRORS, tags=["documents"]
)
def set_access(
    doc_id: str,
    policy: AccessPolicy,
    who: Annotated[Caller, Depends(admin)],
    svc: Annotated[Services, Depends(services)],
) -> AccessOut:
    """Replace a document's permissions (tenant admins only). Nothing is re-embedded.

    The change is saved as an override on top of the policy registry, so a later
    `scripts/ingest.py` run keeps it instead of re-applying the registry's rules.
    """
    info = svc.index.get_document(doc_id, who.principal)
    if info is None:
        raise HTTPException(404, "No such document")
    svc.overrides.set(who.principal.tenant_id, info.source, policy)
    version = svc.index.set_access(doc_id, policy)
    svc.audit.write(
        {
            **_audit_base(who, "PUT /documents/access"),
            "doc_id": doc_id,
            "source": info.source,
            "access_version": version,
            "classification": policy.classification,
            "visibility": policy.visibility,
            "groups": policy.allowed_groups,
            "roles": policy.allowed_roles,
            "users": len(policy.allowed_users),
        }
    )
    return AccessOut(
        doc_id=doc_id,
        source=info.source,
        access_version=version,
        classification=policy.classification,
        visibility=policy.visibility,
        allowed_groups=policy.allowed_groups,
        allowed_roles=policy.allowed_roles,
        allowed_users=policy.allowed_users,
        note="Applied to every chunk without re-embedding; kept as an override of the registry.",
    )


# ----------------------------------------------------------------- ingest


@router.post("/ingest", status_code=202, response_model=JobOut, responses=_ERRORS, tags=["ingest"])
async def ingest(
    who: Annotated[Caller, Depends(admin)],
    svc: Annotated[Services, Depends(services)],
    files: Annotated[list[UploadFile], File(description="PDF files")],
    force: Annotated[bool, Form()] = False,
) -> JobOut:
    """Upload PDFs to index for your tenant (tenant admins only). Returns a job at once;
    follow it with `GET /ingest/{job_id}`. Files with high-risk personal data are quarantined
    (not indexed) and reported in the job."""
    api = svc.settings.api
    if len(files) > api.max_files_per_upload:
        raise HTTPException(422, f"At most {api.max_files_per_upload} files per upload")
    job_id, folder = svc.jobs.new_upload_dir()
    saved: list[tuple[Path, str]] = []
    limit = api.max_upload_mb * 1024 * 1024
    for upload in files:
        name = Path(upload.filename or "").name
        if not name.lower().endswith(".pdf"):
            raise HTTPException(422, f"Only PDF files are accepted, not {name or '(no name)'!r}")
        data = await upload.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"{name} is larger than {api.max_upload_mb} MB")
        if not data.startswith(b"%PDF-"):
            raise HTTPException(422, f"{name} is not a PDF file")
        path = folder / name
        path.write_bytes(data)
        saved.append((path, name))

    pipeline = svc.pipeline(who.principal.tenant_id)
    request_id = request_id_var.get()

    def run_file(path: Path, source: str) -> FileResult:
        outcome = ingest_file(path, svc.index, pipeline, source=source, force=force)
        if outcome.ok and outcome.result:
            r = outcome.result
            return FileResult(source, r.status, r.chunks, None, outcome.warnings)
        status = "quarantined" if outcome.quarantined else "failed"
        return FileResult(source, status, 0, outcome.error, outcome.warnings)

    def on_done(job: Job) -> None:
        svc.audit.write(
            {
                **_audit_base(who, "/ingest"),
                "request_id": request_id,
                "job_id": job.job_id,
                "job_status": job.status,
                "files": [
                    {"source": f.source, "status": f.status, "chunks": f.chunks} for f in job.files
                ],
            }
        )

    job = svc.jobs.submit(
        job_id,
        folder,
        saved,
        tenant_id=who.principal.tenant_id,
        requested_by=who.principal.user_id,
        run_file=run_file,
        on_done=on_done,
    )
    return JobOut(**job.to_dict())


@router.get("/ingest/{job_id}", response_model=JobOut, responses=_ERRORS, tags=["ingest"])
def ingest_status(
    job_id: str,
    who: Annotated[Caller, Depends(admin)],
    svc: Annotated[Services, Depends(services)],
) -> JobOut:
    """Progress of an ingest job, with each file's result."""
    job = svc.jobs.get(job_id, who.principal.tenant_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return JobOut(**job.to_dict())


# ----------------------------------------------------------------- health


@router.get("/health", tags=["health"])
def health(svc: Annotated[Services, Depends(services)]) -> JSONResponse:
    """Service health, for operations. No API key; no tenant data."""
    body: dict[str, Any] = {"status": "ok"}
    try:
        qdrant = svc.index.health()
        body["qdrant"] = {"status": qdrant["status"], "collection": qdrant["collection"]}
    except Exception as exc:
        body["status"] = "unavailable"
        body["qdrant"] = {"status": "unreachable", "error": type(exc).__name__}
    if svc.llm_factory is not None:
        breakers = svc.llm_factory.health()
        body["llm"] = breakers
        if breakers and all(b.get("state") == "open" for b in breakers):
            body["status"] = "degraded"
    return JSONResponse(body, status_code=503 if body["status"] == "unavailable" else 200)
