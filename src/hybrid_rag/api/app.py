"""The FastAPI app: ``uvicorn hybrid_rag.api.app:app`` (or ``create_app()`` in tests)."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from hybrid_rag.access import AccessDeniedError, AccessError
from hybrid_rag.api.logs import request_id_var, setup_logging, tenant_var
from hybrid_rag.api.routes import router
from hybrid_rag.api.services import Services, build_services
from hybrid_rag.config import get_settings
from hybrid_rag.llm import LLMError
from hybrid_rag.store import StoreError
from hybrid_rag.tracing import flush_traces

logger = logging.getLogger("hybrid_rag.api")

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_CODES = {
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    413: "too_large",
    422: "invalid_request",
    503: "unavailable",
}

DESCRIPTION = """\
Ask questions about the HR policies, with answers cited to policy, section, page and effective
date. Every route except `/health` needs an API key (`X-API-Key`; use **Authorize** above),
which maps to a principal: you only ever see, search and change what that principal may.
"""


def _error(status: int, message: str, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "code": code or _CODES.get(status, "error"),
                "message": message,
                "request_id": request_id_var.get(),
            }
        },
        status_code=status,
        headers={"X-Request-ID": request_id_var.get() or ""},
    )


def create_app(
    services: Services | None = None,
    *,
    build: Callable[[], Services] | None = None,
    configure_logging: bool = True,
) -> FastAPI:
    """``services`` is used as given (tests); otherwise they are built at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            app.state.services = services
        else:
            settings = get_settings()
            if configure_logging:
                setup_logging(json_logs=settings.api.json_logs)
            app.state.services = (build or (lambda: build_services(settings)))()
            if settings.api.warmup:
                app.state.services.warm_up()
        logger.info("API ready")
        yield
        app.state.services.jobs.shutdown()
        flush_traces()

    app = FastAPI(
        title="Hybrid RAG: HR policy assistant",
        version="0.1.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex
        request_id_var.set(request_id)
        tenant_var.set(None)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            response = _error(500, "Internal error; see the logs for this request_id", "internal")
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "key": getattr(getattr(request.state, "caller", None), "key_name", None),
            },
        )
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()
        )
        return _error(422, problems or "Invalid request")

    @app.exception_handler(AccessDeniedError)
    async def denied(_: Request, exc: AccessDeniedError) -> JSONResponse:
        return _error(403, str(exc))

    @app.exception_handler(AccessError)
    async def access_error(_: Request, exc: AccessError) -> JSONResponse:
        logger.error("Access configuration error: %s", exc)
        return _error(500, "Access configuration error; see the logs", "internal")

    @app.exception_handler(StoreError)
    async def store_down(_: Request, exc: StoreError) -> JSONResponse:
        logger.error("Vector store error: %s", exc)
        return _error(503, "The document index is unavailable; try again shortly")

    @app.exception_handler(LLMError)
    async def llm_down(_: Request, exc: LLMError) -> JSONResponse:
        logger.error("LLM error: %s", exc)
        return _error(503, "No language model provider is available; try again shortly")

    return app


def __getattr__(name: str):  # ``uvicorn hybrid_rag.api.app:app`` builds the app lazily
    if name == "app":
        return create_app()
    raise AttributeError(name)
