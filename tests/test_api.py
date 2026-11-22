"""The API end to end, with in-memory Qdrant, the hashing encoder, a keyword reranker and
scripted LLMs (no models, no API calls)."""

import json
from datetime import date
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from hybrid_rag.access import AccessOverrides
from hybrid_rag.api import create_app
from hybrid_rag.api.audit import AuditLog
from hybrid_rag.api.auth import KeyStore, add_key
from hybrid_rag.api.jobs import IngestJobs
from hybrid_rag.api.services import Services
from hybrid_rag.config import (
    AnswerSettings,
    ApiSettings,
    ChunkingSettings,
    DocumentsSettings,
    EmbeddingSettings,
    PrivacySettings,
    RerankerSettings,
    RetrievalSettings,
    Settings,
)
from hybrid_rag.embedding import build_embeddings
from hybrid_rag.generation import AnswerService
from hybrid_rag.indexing import ingest_paths
from hybrid_rag.ingestion import make_doc_id
from hybrid_rag.ingestion.parsers import PypdfParser
from hybrid_rag.ingestion.tokens import ApproxTokenCounter
from hybrid_rag.privacy import build_sanitizer
from hybrid_rag.store import DocumentIndex
from tests.llm_fakes import Script, make_failover, openai_server_error
from tests.pdf_fixture import write_pdf
from tests.test_access import ACME_POLICIES, DOCUMENTS, write
from tests.test_retrieval import KeywordCrossEncoder

PRINCIPALS = {
    "employee": {"tenant_id": "acme", "user_id": "u-1", "clearance": "internal"},
    "hr": {
        "tenant_id": "acme",
        "user_id": "u-2",
        "clearance": "confidential",
        "groups": ["hr-compensation"],
    },
    "admin": {"tenant_id": "acme", "user_id": "u-admin", "clearance": "restricted", "admin": True},
    "globex-admin": {
        "tenant_id": "globex",
        "user_id": "u-g",
        "clearance": "restricted",
        "admin": True,
    },
}


class Env:
    def __init__(self, tmp: Path, llm_items: list) -> None:
        registry = tmp / "registry.yaml"
        registry.write_text(yaml.safe_dump({"policies": ACME_POLICIES}), encoding="utf-8")
        self.audit_path = tmp / "audit.jsonl"
        self.overrides_path = tmp / "overrides.json"
        self.settings = Settings.model_construct(
            documents=DocumentsSettings(
                tenant_id="acme",
                registry=registry,
                parser="pypdf",
                cache_dir=None,
                access_overrides=self.overrides_path,
            ),
            chunking=ChunkingSettings(tokenizer="approx"),
            privacy=PrivacySettings(detector="patterns"),
            retrieval=RetrievalSettings(fused_limit=10, top_k=4),
            reranker=RerankerSettings(),
            answer=AnswerSettings(min_rerank_score=None),
            api=ApiSettings(max_upload_mb=1),
        )
        keys_file = tmp / "keys.yaml"
        self.keys = {name: add_key(keys_file, name, spec) for name, spec in PRINCIPALS.items()}
        sanitizer = build_sanitizer(self.settings.privacy)
        index = DocumentIndex(
            QdrantClient(":memory:"), "api", build_embeddings(EmbeddingSettings(provider="hashing"))
        )
        self.script = Script(llm_items)
        llm, _, _ = make_failover(self.script, max_attempts=1)
        answers = AnswerService(
            self.settings,
            index,
            llm,
            sanitizer=sanitizer,
            tokens=ApproxTokenCounter(),
            cross_encoder=KeywordCrossEncoder(),
            today=date(2026, 11, 21),
        )
        self.services = Services(
            settings=self.settings,
            index=index,
            answers=answers,
            keys=KeyStore.load(keys_file),
            audit=AuditLog(self.audit_path),
            jobs=IngestJobs(tmp / "uploads"),
            overrides=AccessOverrides(self.overrides_path),
            sanitizer=sanitizer,
            parser=PypdfParser(),
        )
        docs = tmp / "docs"
        for name, term in DOCUMENTS.items():
            write(docs, name, term)
        self.docs = docs
        assert all(o.ok for o in ingest_paths([docs], index, self.services.pipeline("acme")))

    def headers(self, who: str) -> dict[str, str]:
        return {"X-API-Key": self.keys[who]}

    def audit(self) -> list[dict]:
        return [json.loads(line) for line in self.audit_path.read_text().splitlines()]


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path, ["The answer is in the handbook [1]."] * 20)


@pytest.fixture
def client(env: Env):
    with TestClient(create_app(env.services, configure_logging=False)) as c:
        yield c


# ------------------------------------------------------------------ auth


def test_health_needs_no_key_and_shows_no_tenant_data(client) -> None:
    r = client.get("/health")

    assert r.status_code == 200
    assert r.json()["qdrant"]["status"] and "points" not in r.text


@pytest.mark.parametrize("key", [None, "hrk_wrong"])
def test_routes_need_a_known_key(client, key) -> None:
    headers = {"X-API-Key": key} if key else {}

    r = client.get("/documents", headers=headers)

    assert r.status_code == 401
    error = r.json()["error"]
    assert error["code"] == "unauthorized" and error["request_id"] == r.headers["X-Request-ID"]


def test_a_valid_request_id_is_kept(client, env) -> None:
    r = client.get("/documents", headers={**env.headers("employee"), "X-Request-ID": "trace-123"})

    assert r.headers["X-Request-ID"] == "trace-123"


def test_admin_routes_refuse_other_keys(client, env) -> None:
    doc_id = make_doc_id("acme", "handbook.pdf")

    assert client.delete(f"/documents/{doc_id}", headers=env.headers("hr")).status_code == 403
    r = client.put(f"/documents/{doc_id}/access", json={}, headers=env.headers("employee"))
    assert r.status_code == 403


# ------------------------------------------------------------- documents


def test_documents_are_scoped_to_the_caller(client, env) -> None:
    employee = client.get("/documents", headers=env.headers("employee")).json()
    hr = client.get("/documents", headers=env.headers("hr")).json()
    admin = client.get("/documents", headers=env.headers("admin")).json()

    assert [d["source"] for d in employee] == ["handbook.pdf"]
    assert {d["source"] for d in hr} == {"handbook.pdf", "salary_bands.pdf"}
    assert len(admin) == len(DOCUMENTS)
    assert employee[0]["allowed_groups"] is None  # grants are shown to admins only
    salary = next(d for d in admin if d["source"] == "salary_bands.pdf")
    assert salary["allowed_groups"] == ["hr-compensation"]


def test_delete_is_tenant_scoped_and_hides_what_you_cannot_see(client, env) -> None:
    doc_id = make_doc_id("acme", "manager_guide.pdf")

    other = client.delete(f"/documents/{doc_id}", headers=env.headers("globex-admin"))
    assert other.status_code == 404  # same answer as for a document that doesn't exist

    r = client.delete(f"/documents/{doc_id}", headers=env.headers("admin"))
    assert r.status_code == 200 and r.json()["chunks_removed"] >= 1
    sources = [d["source"] for d in client.get("/documents", headers=env.headers("admin")).json()]
    assert "manager_guide.pdf" not in sources
    assert env.audit()[-1]["route"] == "DELETE /documents"


def test_access_change_applies_without_reembedding_and_survives_reingest(client, env) -> None:
    doc_id = make_doc_id("acme", "salary_bands.pdf")
    body = {"classification": "internal", "visibility": "tenant"}

    r = client.put(f"/documents/{doc_id}/access", json=body, headers=env.headers("admin"))

    assert r.status_code == 200 and r.json()["access_version"] == 2
    employee = client.get("/documents", headers=env.headers("employee")).json()
    assert "salary_bands.pdf" in {d["source"] for d in employee}
    assert json.loads(env.overrides_path.read_text())["acme"]["salary_bands.pdf"]
    # A later CLI-style ingest applies the override, not the registry's restricted rules.
    (outcome,) = [
        o
        for o in ingest_paths([env.docs], env.services.index, env.services.pipeline("acme"))
        if o.path.name == "salary_bands.pdf"
    ]
    assert outcome.result.status == "unchanged"
    assert "salary_bands.pdf" in {
        d["source"] for d in client.get("/documents", headers=env.headers("employee")).json()
    }


def test_invalid_access_rules_are_refused(client, env) -> None:
    doc_id = make_doc_id("acme", "handbook.pdf")

    r = client.put(
        f"/documents/{doc_id}/access", json={"visibilty": "tenant"}, headers=env.headers("admin")
    )

    assert r.status_code == 422 and "visibilty" in r.json()["error"]["message"]


# ----------------------------------------------------------------- query


def test_query_returns_a_cited_answer_and_writes_the_audit_log(client, env) -> None:
    r = client.post(
        "/query",
        json={"question": "QXHANDBOOK rules? Mail me at jane@corp.example"},
        headers=env.headers("employee"),
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "answered" and body["grounded"]
    assert body["citations"][0]["source"] == "handbook.pdf" and body["debug"] is None
    (entry,) = [e for e in env.audit() if e["route"] == "/query"]
    assert entry["request_id"] == body["request_id"] == r.headers["X-Request-ID"]
    assert entry["user_id"] == "u-1" and entry["key"] == "employee"
    assert entry["chunks"][0]["access_version"] == 1 and entry["chunks"][0]["cited"]
    assert entry["redactions"] == {"EMAIL_ADDRESS": 1}
    assert "question" not in entry and len(entry["question_sha256"]) == 64
    assert "jane@corp.example" not in env.audit_path.read_text()


def test_debug_shows_sources_and_never_unreadable_ones(client, env) -> None:
    r = client.post(
        "/query",
        json={"question": "QXSALARY bands", "debug": True},
        headers=env.headers("employee"),
    )

    debug = r.json()["debug"]
    assert debug["sources"] and debug["provider"]["provider"] == "p0"
    assert all(s["source"] != "salary_bands.pdf" for s in debug["sources"])


def test_streaming_sends_tokens_then_the_answer(client, env) -> None:
    with client.stream(
        "POST",
        "/query",
        json={"question": "QXHANDBOOK?", "stream": True},
        headers=env.headers("employee"),
    ) as r:
        text = r.read().decode()

    assert r.headers["content-type"].startswith("text/event-stream")
    events = [block.split("\n") for block in text.strip().split("\n\n")]
    names = [lines[0].removeprefix("event: ") for lines in events]
    assert names[-1] == "answer" and set(names[:-1]) == {"token"}
    final = json.loads(events[-1][1].removeprefix("data: "))
    assert final["grounded"] and final["citations"]


@pytest.mark.parametrize(
    "body",
    [{"question": ""}, {"question": "x" * 2001}, {"question": "ok", "colour": "red"}, {}],
)
def test_invalid_queries_get_a_clear_422(client, env, body) -> None:
    r = client.post("/query", json=body, headers=env.headers("employee"))

    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_llm_outage_is_a_503(tmp_path) -> None:
    env = Env(tmp_path, [openai_server_error()])
    with TestClient(create_app(env.services, configure_logging=False)) as client:
        r = client.post("/query", json={"question": "QXHANDBOOK?"}, headers=env.headers("employee"))

    assert r.status_code == 503 and r.json()["error"]["code"] == "unavailable"


# ---------------------------------------------------------------- ingest


def test_upload_is_indexed_in_the_background(client, env, tmp_path) -> None:
    pdf = write_pdf(tmp_path / "new_policy.pdf", [["1. Gym", "The QXGYM stipend is 50 dollars."]])

    r = client.post(
        "/ingest",
        files=[("files", ("new_policy.pdf", pdf.read_bytes(), "application/pdf"))],
        headers=env.headers("admin"),
    )

    assert r.status_code == 202 and r.json()["status"] in {"queued", "running", "done"}
    env.services.jobs.wait_idle()
    job = client.get(f"/ingest/{r.json()['job_id']}", headers=env.headers("admin")).json()
    assert job["status"] == "done"
    (file,) = job["files"]
    assert file["status"] == "added" and file["chunks"] >= 1
    assert any("not in the policy registry" in w for w in file["warnings"])  # deny by default
    other = client.get(f"/ingest/{r.json()['job_id']}", headers=env.headers("globex-admin"))
    assert other.status_code == 404
    assert any(e["route"] == "/ingest" and e["job_status"] == "done" for e in env.audit())


def test_quarantined_uploads_are_reported(client, env, tmp_path) -> None:
    pdf = write_pdf(tmp_path / "payroll.pdf", [["1. Pay", "SSN 123-45-6789 for the payroll."]])

    r = client.post(
        "/ingest",
        files=[("files", ("payroll.pdf", pdf.read_bytes()))],
        headers=env.headers("admin"),
    )
    env.services.jobs.wait_idle()
    (file,) = client.get(f"/ingest/{r.json()['job_id']}", headers=env.headers("admin")).json()[
        "files"
    ]

    assert file["status"] == "quarantined" and "US_SSN" in file["error"]
    assert "123-45-6789" not in file["error"]


@pytest.mark.parametrize(
    ("name", "data", "status"),
    [
        ("notes.txt", b"%PDF-1.4", 422),
        ("fake.pdf", b"not a pdf", 422),
        ("big.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 1), 413),
    ],
)
def test_bad_uploads_are_refused(client, env, name, data, status) -> None:
    r = client.post("/ingest", files=[("files", (name, data))], headers=env.headers("admin"))

    assert r.status_code == status
