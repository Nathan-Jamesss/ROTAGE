"""Pipeline integration tests, run against real seed data with a fake
extractor standing in for the model — no network, no API key required.

These exercise the full chain: extraction -> matching -> threshold -> ledger,
which is where the vulnerable-donation guard (filter_auto_matchable_reverse)
and the "escalated requests stay open" fix actually matter, as opposed to
threshold.py's unit tests which check each rule in isolation.
"""

from __future__ import annotations

import pytest

from quartermaster import pipeline, store
from quartermaster.schema import (
    ExtractedField,
    Extraction,
    Outcome,
    Tier,
    VulnerableFlag,
)


@pytest.fixture(autouse=True)
def clean_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    store.reset()
    pipeline.reset_extractor()
    yield
    pipeline.reset_extractor()


def _extractor(**fields):
    """A fake extractor: ignores the message, returns fixed fields."""

    def run(message: str) -> Extraction:
        base = Extraction(
            category=ExtractedField(value=fields.get("category"), confidence=0.9),
            item=ExtractedField(value=fields.get("item", ""), confidence=0.9),
            quantity=ExtractedField(value=fields.get("quantity", 1), confidence=0.9),
            location=ExtractedField(value=fields.get("location", ""), confidence=0.9),
            requester_name=ExtractedField(value="Test Requester", confidence=0.9),
            vulnerable_flags=fields.get("vulnerable_flags", []),
        )
        return base

    return run


def test_clean_need_auto_resolves_against_real_pool() -> None:
    decision = pipeline.process_need(
        "Meera Pillai",
        "need a wheelchair, kaloor",
        extractor=_extractor(category="mobility_aid", item="wheelchair", quantity=1, location="Kaloor"),
    )
    assert decision.outcome == Outcome.AUTO_RESOLVED
    assert decision.chosen == "r-001"

    resources = store.load_resources()
    reserved = next(r for r in resources if r.id == "r-001")
    assert reserved.reserved_for == decision.request_id

    requests = store.load_requests()
    request = next(r for r in requests if r.id == decision.request_id)
    assert request.status == "matched"


def test_vulnerable_need_escalates_despite_perfect_match() -> None:
    decision = pipeline.process_need(
        "Deepa Nair",
        "need a wheelchair, mother is 78 years old and just had hip surgery, kaloor",
        extractor=_extractor(category="mobility_aid", item="wheelchair", quantity=1, location="Kaloor"),
    )
    assert decision.outcome == Outcome.ESCALATED
    assert decision.tier == Tier.SAFETY_HOLD

    resources = store.load_resources()
    untouched = next(r for r in resources if r.id == "r-001")
    assert untouched.reserved_for is None, "a vulnerable case must not reserve stock"


def test_escalated_need_stays_open_for_a_future_donation() -> None:
    """The bug this test guards: an escalated request used to be marked
    "escalated" and silently dropped out of the pool a later donation
    searches. If it isn't "open", this whole test is dead on arrival."""
    need_decision = pipeline.process_need(
        "Manoj Pillai",
        "need a standing frame for cerebral palsy, muvattupuzha",
        extractor=_extractor(category="other", item="standing frame for cerebral palsy", quantity=1, location="Muvattupuzha"),
    )
    assert need_decision.outcome == Outcome.ESCALATED  # nothing in seed pool matches

    requests = store.load_requests()
    request = next(r for r in requests if r.id == need_decision.request_id)
    assert request.status == "open"

    donation_decision = pipeline.process_donation(
        "Anonymous Donor",
        "donating a standing frame, muvattupuzha",
        extractor=_extractor(category="other", item="standing frame for cerebral palsy", quantity=1, location="Muvattupuzha"),
    )
    assert donation_decision.outcome == Outcome.AUTO_RESOLVED
    assert donation_decision.chosen is not None

    requests = store.load_requests()
    request = next(r for r in requests if r.id == need_decision.request_id)
    assert request.status == "matched"


def test_donation_withholds_link_to_a_vulnerable_need() -> None:
    """A donation must not auto-link to a need that was flagged vulnerable,
    even though the need is still 'open' and would otherwise score highly."""
    need_decision = pipeline.process_need(
        "Grandmother's Neighbour",
        "hearing aid needed, my grandmother can't hear the doorbell, muvattupuzha",
        extractor=_extractor(
            category="assistive_tech", item="hearing aid", quantity=1,
            location="Muvattupuzha", vulnerable_flags=[VulnerableFlag.ELDERLY],
        ),
    )
    assert need_decision.tier == Tier.SAFETY_HOLD

    donation_decision = pipeline.process_donation(
        "Anonymous Donor",
        "donating a hearing aid, muvattupuzha",
        extractor=_extractor(category="assistive_tech", item="hearing aid", quantity=1, location="Muvattupuzha"),
    )

    assert donation_decision.outcome == Outcome.ESCALATED
    assert donation_decision.tier == Tier.SAFETY_HOLD
    assert donation_decision.chosen is None

    requests = store.load_requests()
    request = next(r for r in requests if r.id == need_decision.request_id)
    assert request.status == "open", "the vulnerable need must not be silently matched"


def test_donation_with_no_open_need_just_joins_the_pool() -> None:
    decision = pipeline.process_donation(
        "Rotary Club of Kochi United",
        "donating 3 more nebulizers, kaloor",
        extractor=_extractor(category="respiratory", item="nebulizer", quantity=3, location="Kaloor"),
    )
    assert decision.outcome == Outcome.AUTO_RESOLVED
    assert decision.chosen is None  # nothing to link to, just added to stock

    resources = store.load_resources()
    assert any(r.item == "nebulizer" and r.donor == "Rotary Club of Kochi United" for r in resources)


def test_shift_signup_assigns_when_slot_is_open() -> None:
    decision = pipeline.process_shift_signup(
        "Ligi Thomas", "i can do first aid saturday morning, i'm a staff nurse", "s-003"
    )
    assert decision.outcome == Outcome.AUTO_RESOLVED
    shifts = store.load_shifts()
    shift = next(s for s in shifts if s.id == "s-003")
    assert "Ligi Thomas" in shift.assigned


def test_shift_signup_escalates_on_double_booking() -> None:
    pipeline.process_shift_signup("Ligi Thomas", "first aid, nurse", "s-003")
    second = pipeline.process_shift_signup(
        "Prasanna Nair", "first aid duty too, basic first aid training", "s-003"
    )
    assert second.outcome == Outcome.ESCALATED
    assert second.tier == Tier.CONTESTED


def test_shift_signup_escalates_on_already_full_slot() -> None:
    """s-004 (Photography) is seeded already at capacity."""
    decision = pipeline.process_shift_signup(
        "New Volunteer", "i can take photos", "s-004"
    )
    assert decision.outcome == Outcome.ESCALATED
    assert decision.tier == Tier.CONTESTED


def test_shift_signup_escalates_on_missing_skill() -> None:
    decision = pipeline.process_shift_signup(
        "Someone Random", "happy to help out on saturday", "s-006"
    )
    assert decision.outcome == Outcome.ESCALATED
    assert "translation" in decision.escalation_reason.lower()


def test_unknown_shift_raises() -> None:
    with pytest.raises(KeyError):
        pipeline.process_shift_signup("Anyone", "message", "s-999")
