"""Access control: the access model, the registry's access rules and leakage tests.

The leakage tests index documents for two tenants with every kind of permission into an
in-memory Qdrant, then check for each caller that search (including an exact-match query for
a forbidden document's unique term), list, get, count and delete return only what the access
rule allows. ``Principal.can_read`` (the rule in Python) is cross-checked against the filter.
"""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from hybrid_rag.access import (
    CLASSIFICATION_RANKS,
    AccessDeniedError,
    AccessPolicy,
    Principal,
)
from hybrid_rag.config import ChunkingSettings, DocumentsSettings, EmbeddingSettings
from hybrid_rag.embedding import build_embeddings
from hybrid_rag.indexing import ingest_paths
from hybrid_rag.ingestion import IngestionPipeline, PolicyRegistry, RegistryError, make_doc_id
from hybrid_rag.store import DocumentIndex
from tests.pdf_fixture import write_pdf

PDFS = Path(__file__).resolve().parents[1] / "data" / "pdfs"
CHUNKING = ChunkingSettings(target_tokens=60, max_tokens=120, overlap_tokens=12, tokenizer="approx")

# Each document has a unique term, so an exact-match (sparse) query targets it directly.
DOCUMENTS = {
    "handbook.pdf": "QXHANDBOOK",
    "salary_bands.pdf": "QXSALARY",
    "manager_guide.pdf": "QXMANAGER",
    "personal_file.pdf": "QXPERSONAL",
    "investigations.pdf": "QXINVESTIGATE",
    "unlisted.pdf": "QXUNLISTED",
}
ACME_POLICIES = {
    "handbook.pdf": {
        "policy_name": "Handbook",
        "access": {"classification": "internal", "visibility": "tenant"},
    },
    "salary_bands.pdf": {
        "policy_name": "Salary Bands",
        "access": {
            "classification": "confidential",
            "allowed_groups": ["hr-compensation"],
            "allowed_roles": ["hr_director"],
        },
    },
    "manager_guide.pdf": {
        "policy_name": "Manager Guide",
        "access": {"classification": "internal", "allowed_roles": ["people_manager"]},
    },
    "personal_file.pdf": {
        "policy_name": "Personal File",
        "access": {"classification": "confidential", "allowed_users": ["u-1042"]},
    },
    "investigations.pdf": {
        "policy_name": "Investigations",
        "access": {"classification": "restricted", "allowed_groups": ["hr-employee-relations"]},
    },
    # unlisted.pdf: no entry and no default access block, so deny by default.
}

EMPLOYEE = Principal("acme", "u-1", "internal")
HR_COMP = Principal("acme", "u-2", "confidential", groups={"hr-compensation"})
HR_COMP_LOW_CLEARANCE = Principal("acme", "u-3", "internal", groups={"hr-compensation"})
HR_DIRECTOR = Principal("acme", "u-4", "confidential", roles={"hr_director"})
MANAGER = Principal("acme", "u-5", "internal", roles={"people_manager"})
OWNER = Principal("acme", "u-1042", "confidential")
OWNER_LOW_CLEARANCE = Principal("acme", "u-1042", "internal")
INVESTIGATOR = Principal("acme", "u-6", "restricted", groups={"hr-employee-relations"})
CLEARED_NO_GRANTS = Principal("acme", "u-7", "restricted")
ACME_ADMIN = Principal.tenant_admin("acme")
GLOBEX_EMPLOYEE = Principal("globex", "u-1", "restricted", groups={"hr-compensation"})
GLOBEX_ADMIN = Principal.tenant_admin("globex")

EXPECTED = [
    (EMPLOYEE, {"handbook.pdf"}),
    (HR_COMP, {"handbook.pdf", "salary_bands.pdf"}),
    (HR_COMP_LOW_CLEARANCE, {"handbook.pdf"}),  # a grant never lifts the classification ceiling
    (HR_DIRECTOR, {"handbook.pdf", "salary_bands.pdf"}),
    (MANAGER, {"handbook.pdf", "manager_guide.pdf"}),
    (OWNER, {"handbook.pdf", "personal_file.pdf"}),
    (OWNER_LOW_CLEARANCE, {"handbook.pdf"}),
    (INVESTIGATOR, {"handbook.pdf", "investigations.pdf"}),
    (CLEARED_NO_GRANTS, {"handbook.pdf"}),  # clearance alone grants nothing
    (ACME_ADMIN, set(DOCUMENTS)),  # including the unlisted, deny-by-default document
    (GLOBEX_EMPLOYEE, {"handbook.pdf"}),  # globex's own handbook, never acme's
    (GLOBEX_ADMIN, {"handbook.pdf"}),
]
IDS = [
    "employee",
    "hr-comp",
    "hr-comp-low-clearance",
    "hr-director",
    "manager",
    "owner",
    "owner-low-clearance",
    "investigator",
    "cleared-no-grants",
    "acme-admin",
    "globex-employee",
    "globex-admin",
]


def write(folder: Path, name: str, term: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{name} policy. The code word is {term}.",
        *[f"Rule {i} of {term}." for i in range(8)],
    ]
    return write_pdf(folder / name, [lines])


def acme_pipeline(policies: dict | None = None) -> IngestionPipeline:
    registry = PolicyRegistry(policies or ACME_POLICIES)
    return IngestionPipeline(CHUNKING, registry=registry, tenant_id="acme")


def globex_pipeline() -> IngestionPipeline:
    registry = PolicyRegistry(
        {"handbook.pdf": {"policy_name": "Globex Handbook", "access": {"visibility": "tenant"}}},
        defaults={"access": {"classification": "internal"}},
    )
    return IngestionPipeline(CHUNKING, registry=registry, tenant_id="globex")


def index_world(index: DocumentIndex, root: Path) -> None:
    """Index the acme documents and globex's handbook. Also used by the live Qdrant test."""
    for name, term in DOCUMENTS.items():
        write(root / "acme", name, term)
    write(root / "globex", "handbook.pdf", "QXGLOBEX")  # same file name, other tenant
    acme = ingest_paths([root / "acme"], index, acme_pipeline())
    globex = ingest_paths([root / "globex"], index, globex_pipeline())
    assert all(o.ok for o in [*acme, *globex])


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> tuple[DocumentIndex, Path]:
    root = tmp_path_factory.mktemp("access")
    index = DocumentIndex(
        QdrantClient(":memory:"), "access", build_embeddings(EmbeddingSettings(provider="hashing"))
    )
    index_world(index, root)
    return index, root


def acme_id(source: str) -> str:
    return make_doc_id("acme", source)


# ---------------------------------------------------------------- leakage


@pytest.mark.parametrize(("principal", "visible"), EXPECTED, ids=IDS)
def test_list_shows_only_readable_documents(world, principal: Principal, visible: set) -> None:
    index, _ = world
    documents = index.list_documents(principal)

    assert {d.source for d in documents} == visible
    assert {d.tenant_id for d in documents} == {principal.tenant_id}


@pytest.mark.parametrize(("principal", "visible"), EXPECTED, ids=IDS)
def test_get_and_count_hide_unreadable_documents(world, principal, visible) -> None:
    index, _ = world
    for source in DOCUMENTS:
        readable = principal.tenant_id == "acme" and source in visible
        info = index.get_document(acme_id(source), principal)

        assert (info is not None) == readable, source
        assert (index.count_chunks(principal, acme_id(source)) > 0) == readable, source


@pytest.mark.parametrize(("principal", "visible"), EXPECTED, ids=IDS)
def test_exact_match_search_never_returns_unreadable_chunks(world, principal, visible) -> None:
    index, _ = world
    for source, term in DOCUMENTS.items():
        hits = index.search(term, principal, k=50)

        assert hits, term  # the readable handbook always matches something
        assert all(principal.can_read(h.metadata) for h in hits)
        assert {h.metadata["source"] for h in hits} <= visible
        assert {h.metadata["tenant_id"] for h in hits} == {principal.tenant_id}
        if principal.tenant_id == "acme" and source in visible:
            assert hits[0].metadata["source"] == source, term  # and the right one ranks first
        else:
            assert all(term not in h.page_content for h in hits), term


def test_filter_agrees_with_can_read_on_every_chunk(world) -> None:
    index, _ = world
    every_chunk = index.search("policy rule code word", ACME_ADMIN, k=500)
    every_chunk += index.search("policy rule code word", GLOBEX_ADMIN, k=500)

    for principal, _ in EXPECTED:
        allowed = {h.id for h in index.search("policy rule code word", principal, k=500)}
        expected = {c.id for c in every_chunk if principal.can_read(c.metadata)}
        assert allowed == expected, principal.user_id


def test_same_file_name_in_two_tenants_gets_two_documents(world) -> None:
    index, _ = world
    acme = index.get_document(acme_id("handbook.pdf"), ACME_ADMIN)
    globex = index.get_document(make_doc_id("globex", "handbook.pdf"), GLOBEX_ADMIN)

    assert acme.doc_id != globex.doc_id
    assert (acme.title, globex.title) == ("Handbook", "Globex Handbook")
    assert index.get_document(globex.doc_id, ACME_ADMIN) is None


def test_relevance_conditions_never_widen_the_security_filter(world) -> None:
    from hybrid_rag.store.schema import match

    index, _ = world
    hits = index.search(
        "QXSALARY", EMPLOYEE, k=50, conditions=[match("source", "salary_bands.pdf")]
    )

    assert hits == []


def test_only_tenant_admins_delete_and_only_in_their_tenant(world) -> None:
    index, _ = world
    with pytest.raises(AccessDeniedError):
        index.delete(acme_id("handbook.pdf"), HR_COMP)

    assert index.delete(acme_id("handbook.pdf"), GLOBEX_ADMIN) == 0
    assert index.get_document(acme_id("handbook.pdf"), EMPLOYEE) is not None


def test_access_fields_are_on_every_chunk_and_never_in_the_text(world) -> None:
    index, _ = world
    for hit in index.search("policy rule code word", ACME_ADMIN, k=500):
        m = hit.metadata
        assert m["tenant_id"] == "acme"
        assert m["classification_rank"] == CLASSIFICATION_RANKS[m["classification"]]
        assert m["access_version"] == 1 and m["access_hash"] and m["access_updated_at"]
        for grant in ("hr-compensation", "hr_director", "people_manager", "u-1042", "restricted"):
            assert grant not in hit.page_content


def test_unlisted_document_is_denied_by_default(world) -> None:
    index, _ = world
    info = index.get_document(acme_id("unlisted.pdf"), ACME_ADMIN)

    assert (info.classification, info.visibility) == ("restricted", "restricted")
    assert info.allowed_groups == info.allowed_roles == info.allowed_users == []


# ------------------------------------------------------ permission changes


def test_permission_change_updates_chunks_without_loading_or_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, term in DOCUMENTS.items():
        write(tmp_path, name, term)
    index = DocumentIndex(
        QdrantClient(":memory:"), "t", build_embeddings(EmbeddingSettings(provider="hashing"))
    )
    ingest_paths([tmp_path], index, acme_pipeline())
    salary = acme_id("salary_bands.pdf")
    before = index.get_document(salary, ACME_ADMIN)
    assert index.get_document(salary, EMPLOYEE) is None

    opened = {**ACME_POLICIES, "salary_bands.pdf": {**ACME_POLICIES["salary_bands.pdf"]}}
    opened["salary_bands.pdf"]["access"] = {"classification": "internal", "visibility": "tenant"}
    changed = acme_pipeline(opened)
    assert changed.fingerprint("salary_bands.pdf") == acme_pipeline().fingerprint(
        "salary_bands.pdf"
    )  # access isn't part of the content fingerprint

    def fail(*_args, **_kwargs):
        raise AssertionError("nothing should be loaded or embedded")

    monkeypatch.setattr(changed, "load", fail)
    monkeypatch.setattr(index, "_write_chunks", fail)
    outcomes = {o.path.name: o.result for o in ingest_paths([tmp_path], index, changed)}

    assert outcomes["salary_bands.pdf"].status == "access_updated"
    assert outcomes["salary_bands.pdf"].access_version == 2
    assert {o.status for n, o in outcomes.items() if n != "salary_bands.pdf"} == {"unchanged"}
    after = index.get_document(salary, EMPLOYEE)
    assert after is not None
    assert (after.classification, after.visibility, after.access_version) == (
        "internal",
        "tenant",
        2,
    )
    assert after.access_hash != before.access_hash
    assert index.count_chunks(EMPLOYEE, salary) == before.chunk_count  # every chunk updated
    assert after.ingested_at == before.ingested_at  # the content was not rewritten


def test_set_access_can_revoke_and_keeps_the_tenant(tmp_path: Path) -> None:
    index = DocumentIndex(
        QdrantClient(":memory:"), "t", build_embeddings(EmbeddingSettings(provider="hashing"))
    )
    ingest_paths([write(tmp_path, "handbook.pdf", "QXHANDBOOK")], index, acme_pipeline())
    doc_id = acme_id("handbook.pdf")

    assert index.set_access(doc_id, AccessPolicy(classification="internal")) == 2

    assert index.get_document(doc_id, EMPLOYEE) is None  # no grants left
    assert index.search("QXHANDBOOK", EMPLOYEE, k=10) == []
    assert index.get_document(doc_id, ACME_ADMIN).tenant_id == "acme"


def test_content_change_keeps_access_version_unless_access_changed(tmp_path: Path) -> None:
    index = DocumentIndex(
        QdrantClient(":memory:"), "t", build_embeddings(EmbeddingSettings(provider="hashing"))
    )
    path = write(tmp_path, "handbook.pdf", "QXHANDBOOK")
    ingest_paths([path], index, acme_pipeline())
    write(tmp_path, "handbook.pdf", "QXCHANGED")

    (outcome,) = ingest_paths([path], index, acme_pipeline())

    assert (outcome.result.status, outcome.result.access_version) == ("updated", 1)


# ------------------------------------------------------------ the model


def test_access_policy_defaults_are_closed() -> None:
    policy = AccessPolicy()

    assert (policy.classification, policy.visibility, policy.has_readers) == (
        "restricted",
        "restricted",
        False,
    )


def test_access_policy_normalises_and_validates_grants() -> None:
    policy = AccessPolicy(allowed_groups=["b", " a ", "b"], allowed_roles="hr_director")

    assert (policy.allowed_groups, policy.allowed_roles) == (["a", "b"], ["hr_director"])
    assert (
        policy.hash == AccessPolicy(allowed_groups=["a", "b"], allowed_roles=["hr_director"]).hash
    )
    with pytest.raises(ValueError, match="no spaces"):
        AccessPolicy(allowed_groups=["hr compensation"])
    with pytest.raises(ValueError, match="allowed_group"):
        AccessPolicy(allowed_group=["x"])  # a typo must not become a silent deny or grant


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tenant_id": "", "user_id": "u", "clearance": "internal"},
        {"tenant_id": "Acme Corp", "user_id": "u", "clearance": "internal"},
        {"tenant_id": "acme", "user_id": " ", "clearance": "internal"},
        {"tenant_id": "acme", "user_id": "u", "clearance": "top-secret"},
    ],
)
def test_incomplete_principals_are_refused(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Principal(**kwargs)


def test_tenant_id_setting_is_validated() -> None:
    with pytest.raises(ValueError, match="tenant"):
        DocumentsSettings(tenant_id="Not Valid")


def test_tenant_scoped_doc_ids() -> None:
    assert make_doc_id("acme", "a.pdf") == make_doc_id("acme", "a.pdf")
    assert make_doc_id("acme", "a.pdf") != make_doc_id("globex", "a.pdf")


# --------------------------------------------------------------- registry


def test_registry_merges_default_access_key_by_key() -> None:
    registry = PolicyRegistry(
        {"s.pdf": {"policy_name": "S", "access": {"classification": "confidential"}}},
        defaults={"access": {"classification": "internal", "visibility": "tenant"}},
    )

    policy, warnings = registry.access("s.pdf")
    assert (policy.classification, policy.visibility) == ("confidential", "tenant")
    assert warnings == []
    assert registry.access("other.pdf")[0].classification == "internal"


def test_registry_warns_about_documents_nobody_can_read() -> None:
    registry = PolicyRegistry(
        {"s.pdf": {"policy_name": "S", "access": {"visibility": "restricted"}}}
    )

    _, warnings = registry.access("s.pdf")

    assert any("only tenant admins" in w for w in warnings)


@pytest.mark.parametrize(
    "access",
    [
        {"classification": "secret"},
        {"visibility": "everyone"},
        {"allowed_group": ["x"]},
        ["not", "a", "mapping"],
    ],
)
def test_registry_rejects_invalid_access_rules(access) -> None:
    with pytest.raises(RegistryError, match="access rules"):
        PolicyRegistry({"s.pdf": {"policy_name": "S", "access": access}})


def test_access_change_does_not_change_the_entry_hash() -> None:
    a = PolicyRegistry({"s.pdf": {"policy_name": "S", "access": {"visibility": "tenant"}}})
    b = PolicyRegistry({"s.pdf": {"policy_name": "S", "access": {"visibility": "restricted"}}})

    assert a.entry_hash("s.pdf") == b.entry_hash("s.pdf")
    assert a.access("s.pdf")[0].hash != b.access("s.pdf")[0].hash
    assert "access" not in a.lookup("s.pdf", title="S")[0].metadata()


def test_hr_registry_restricts_only_the_compensation_policy() -> None:
    registry = PolicyRegistry.load(PDFS / "registry.yaml")
    access = {pdf.name: registry.access(pdf.name)[0] for pdf in PDFS.glob("*.pdf")}

    merit = access.pop("Annual_Review_Merit_Bonus_Policy_2027.pdf")
    assert (merit.classification, merit.visibility) == ("confidential", "restricted")
    assert merit.allowed_groups == ["hr-compensation"]
    assert merit.allowed_roles == ["hr_director", "people_manager"]
    assert {(a.classification, a.visibility) for a in access.values()} == {("internal", "tenant")}
