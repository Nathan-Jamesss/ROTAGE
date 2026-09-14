"""Ledger: decisions are created, actioned, and undone without ever being
deleted or silently rewritten."""

from __future__ import annotations

import pytest

from quartermaster import ledger, store
from quartermaster.schema import (
    Candidate,
    ExtractedField,
    Extraction,
    HumanAction,
    IntakeKind,
    Outcome,
    RuleResult,
    Tier,
)


@pytest.fixture(autouse=True)
def clean_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    store.reset()
    yield


def _extraction() -> Extraction:
    return Extraction(
        category=ExtractedField(value="mobility_aid", confidence=0.9),
        item=ExtractedField(value="wheelchair", confidence=0.9),
        quantity=ExtractedField(value=1, confidence=0.9),
        requester_name=ExtractedField(value="Meera Pillai", confidence=0.95),
    )


def _rules() -> list[RuleResult]:
    return [RuleResult(rule="T1_vulnerable_person", passed=True, statement="none")]


def test_create_persists_and_returns_a_decision() -> None:
    decision = ledger.create(
        request_id="q-001",
        kind=IntakeKind.NEED,
        raw_message="need a wheelchair",
        requester="Meera Pillai",
        extraction=_extraction(),
        candidates=[Candidate(resource_id="r-001", score=95.0)],
        chosen="r-001",
        rules=_rules(),
        outcome=Outcome.AUTO_RESOLVED,
        tier=Tier.AUTO_RESOLVED,
        confidence=0.9,
        escalation_reason=None,
    )
    assert decision.id.startswith("d-")
    stored = store.get_decision(decision.id)
    assert stored is not None
    assert stored.chosen == "r-001"


def test_auto_resolved_with_a_choice_is_reversible() -> None:
    decision = ledger.create(
        request_id="q-001", kind=IntakeKind.NEED, raw_message="x", requester="x",
        extraction=_extraction(), candidates=[], chosen="r-001", rules=_rules(),
        outcome=Outcome.AUTO_RESOLVED, tier=Tier.AUTO_RESOLVED, confidence=0.9,
        escalation_reason=None,
    )
    assert decision.reversible is True


def test_escalated_decision_is_not_reversible() -> None:
    decision = ledger.create(
        request_id="q-001", kind=IntakeKind.NEED, raw_message="x", requester="x",
        extraction=_extraction(), candidates=[], chosen=None, rules=_rules(),
        outcome=Outcome.ESCALATED, tier=Tier.SAFETY_HOLD, confidence=0.9,
        escalation_reason="vulnerable",
    )
    assert decision.reversible is False
    with pytest.raises(ValueError):
        ledger.undo(decision.id)


def test_undo_releases_the_reservation_and_marks_reversed() -> None:
    resources = store.load_resources()
    target = resources[0]
    store.reserve_resource(target.id, "q-001")

    decision = ledger.create(
        request_id="q-001", kind=IntakeKind.NEED, raw_message="x", requester="x",
        extraction=_extraction(), candidates=[], chosen=target.id, rules=_rules(),
        outcome=Outcome.AUTO_RESOLVED, tier=Tier.AUTO_RESOLVED, confidence=0.9,
        escalation_reason=None,
    )

    reversed_decision = ledger.undo(decision.id)

    assert reversed_decision.outcome == Outcome.REVERSED
    assert reversed_decision.reversed_at is not None
    refreshed = next(r for r in store.load_resources() if r.id == target.id)
    assert refreshed.reserved_for is None


def test_undo_twice_raises() -> None:
    decision = ledger.create(
        request_id="q-001", kind=IntakeKind.NEED, raw_message="x", requester="x",
        extraction=_extraction(), candidates=[], chosen="r-001", rules=_rules(),
        outcome=Outcome.AUTO_RESOLVED, tier=Tier.AUTO_RESOLVED, confidence=0.9,
        escalation_reason=None,
    )
    ledger.undo(decision.id)
    with pytest.raises(ValueError):
        ledger.undo(decision.id)


def test_original_record_survives_undo_unmodified() -> None:
    """Undo must not erase what the decision originally said."""
    decision = ledger.create(
        request_id="q-001", kind=IntakeKind.NEED, raw_message="need a wheelchair",
        requester="Meera Pillai", extraction=_extraction(), candidates=[],
        chosen="r-001", rules=_rules(), outcome=Outcome.AUTO_RESOLVED,
        tier=Tier.AUTO_RESOLVED, confidence=0.9, escalation_reason=None,
    )
    ledger.undo(decision.id)
    reversed_decision = store.get_decision(decision.id)
    assert reversed_decision.raw_message == "need a wheelchair"
    assert reversed_decision.chosen == "r-001"


def test_human_action_is_recorded() -> None:
    decision = ledger.create(
        request_id="q-001", kind=IntakeKind.NEED, raw_message="x", requester="x",
        extraction=_extraction(), candidates=[], chosen=None, rules=_rules(),
        outcome=Outcome.ESCALATED, tier=Tier.CONTESTED, confidence=0.8,
        escalation_reason="tie",
    )
    updated = ledger.apply_human_action(decision.id, HumanAction.ACCEPT, "go ahead")
    assert updated.human_action == HumanAction.ACCEPT
    assert updated.human_note == "go ahead"


def test_receipt_includes_message_rules_and_outcome() -> None:
    decision = ledger.create(
        request_id="q-001", kind=IntakeKind.NEED,
        raw_message="need a wheelchair, mother is 78 and had surgery",
        requester="Deepa Nair", extraction=_extraction(), candidates=[],
        chosen=None, rules=_rules(), outcome=Outcome.ESCALATED,
        tier=Tier.SAFETY_HOLD, confidence=0.9,
        escalation_reason="Elderly + medical mention.",
    )
    text = ledger.render_receipt(decision)
    assert "need a wheelchair" in text
    assert "Deepa Nair" in text
    assert "T1_vulnerable_person" in text
    assert "SAFETY_HOLD" in text or "1_safety_hold" in text
    assert "Elderly" in text


def test_acting_on_unknown_decision_raises() -> None:
    with pytest.raises(KeyError):
        ledger.undo("d-9999")
    with pytest.raises(KeyError):
        ledger.apply_human_action("d-9999", HumanAction.IGNORE)
