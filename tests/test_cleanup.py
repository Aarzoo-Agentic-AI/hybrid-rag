"""Repairs to parsed blocks: contents spill, continuations and registry corrections."""

import pytest

from hybrid_rag.ingestion import PolicyRegistry, RegistryError
from hybrid_rag.ingestion.blocks import Block
from hybrid_rag.ingestion.cleanup import Correction, repair_blocks


def heading(text: str, page: int = 1) -> Block:
    return Block("heading", text, page)


def para(text: str, page: int = 1) -> Block:
    return Block("paragraph", text, page)


def item(text: str, page: int = 1) -> Block:
    return Block("list_item", text, page)


SECTIONS = [heading("7. Travel and Expenses"), heading("8. Corrective Action")]
SECTIONS += [heading("9. Separation from Employment")]


def texts(blocks: list[Block]) -> list[str]:
    return [b.text for b in blocks]


def test_misplaced_contents_entries_are_dropped_and_the_sentence_rejoined() -> None:
    blocks = [
        heading("2. Equal Opportunity"),
        para("We prohibit discrimination based on age, disability,"),
        item("Travel and expenses"),
        item("Corrective action"),
        item("Separation from employment"),
        para("marital status, or any other protected characteristic."),
        *SECTIONS,
    ]

    repaired, warnings = repair_blocks(blocks)

    assert texts(repaired)[:2] == [
        "2. Equal Opportunity",
        "We prohibit discrimination based on age, disability, marital status, or any other "
        "protected characteristic.",
    ]
    assert warnings == []


def test_short_lists_and_the_real_contents_page_are_kept() -> None:
    real_contents = [heading("Contents"), *(item(t) for t in ("Travel and expenses",) * 3)]
    genuine = [heading("1. Purpose"), item("Travel and expenses"), item("Corrective action")]

    repaired, _ = repair_blocks([*real_contents, *genuine, *SECTIONS])

    assert len(repaired) == len(real_contents) + len(genuine) + len(SECTIONS)


def test_a_new_sentence_or_list_is_not_joined() -> None:
    blocks = [para("Ends with a full stop."), para("lower case after a full stop"), para("Ends:")]
    blocks += [para("lower case after a colon"), para("No punctuation"), para("New sentence.")]

    repaired, _ = repair_blocks(blocks)

    assert len(repaired) == len(blocks)


def test_corrections_are_applied_and_stale_ones_reported() -> None:
    blocks = [para("Acknowledgment form marital status, or any other.")]
    fixes = [Correction(find="Acknowledgment form ", replace=""), Correction(find="not there")]

    repaired, warnings = repair_blocks(blocks, fixes)

    assert texts(repaired) == ["marital status, or any other."]
    assert warnings == ["registry correction not applied, text not found: 'not there'"]
    assert texts(blocks) == ["Acknowledgment form marital status, or any other."]  # unchanged


def test_registry_corrections_are_validated_and_part_of_the_entry_hash() -> None:
    fix = {"find": "Acknowledgment form", "replace": ""}
    plain = PolicyRegistry({"h.pdf": {"policy_name": "H"}})
    fixed = PolicyRegistry({"h.pdf": {"policy_name": "H", "corrections": [fix]}})

    assert fixed.corrections("h.pdf") == [Correction(**fix)]
    assert plain.corrections("h.pdf") == []
    assert plain.entry_hash("h.pdf") != fixed.entry_hash("h.pdf")  # re-processes the file
    assert "corrections" not in fixed.lookup("h.pdf", title="H")[0].metadata()
    with pytest.raises(RegistryError, match="corrections"):
        PolicyRegistry({"h.pdf": {"policy_name": "H", "corrections": [{"find": "x"}]}})
    with pytest.raises(RegistryError, match="corrections"):
        PolicyRegistry({"h.pdf": {"policy_name": "H", "corrections": [{"fnd": "abc"}]}})
