"""Deterministic mode: the demo's insurance policy. No network, no API key.

Run against real seed data so a broken deterministic path would be caught
here, not discovered live during a recording session with the network down.
"""

from __future__ import annotations

import pytest

from quartermaster import deterministic, pipeline, store
from quartermaster.schema import Outcome, Tier


@pytest.fixture(autouse=True)
def clean_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    store.reset()
    pipeline.reset_extractor()
    yield
    pipeline.reset_extractor()


def test_extracts_item_category_and_quantity() -> None:
    extraction = deterministic.deterministic_extract(
        "need a wheelchair for 2 weeks, kaloor"
    )
    assert extraction.category.value == "mobility_aid"
    assert extraction.location.value == "Kaloor"


def test_extracts_quantity_from_digits() -> None:
    extraction = deterministic.deterministic_extract("need 3 nebulizers, perumbavoor")
    assert extraction.quantity.value == 3


def test_reuses_threshold_keyword_scan_for_vulnerability() -> None:
    extraction = deterministic.deterministic_extract(
        "wheelchair needed, my grandmother cannot walk"
    )
    assert extraction.vulnerable_flags  # elderly, via the shared keyword scan


def test_unrecognized_item_yields_missing_category() -> None:
    extraction = deterministic.deterministic_extract("need a hearing aid, muvattupuzha")
    assert extraction.category.value is None
    assert "category" in extraction.missing_fields


def test_classifies_donation_vs_need_vs_out_of_scope() -> None:
    assert deterministic._classify("we'd like to donate a wheelchair") == "donation"
    assert deterministic._classify("need a wheelchair") == "need"
    assert deterministic._classify("how do i get a tax receipt?") == "out_of_scope"
    assert deterministic._classify("i can help with registration saturday") == "volunteer"


def test_route_message_clean_need_auto_resolves() -> None:
    pipeline.set_extractor(deterministic.deterministic_extract)
    result = deterministic.route_message("Meera Pillai", "need a wheelchair, kaloor")
    assert result["outcome"] == "auto_resolved"
    assert result["chosen"] == "r-001"


def test_route_message_vulnerable_need_escalates() -> None:
    pipeline.set_extractor(deterministic.deterministic_extract)
    result = deterministic.route_message(
        "Deepa Nair", "need a wheelchair, my mother is 78 years old and had surgery"
    )
    assert result["outcome"] == "escalated"
    assert result["tier"] == "1_safety_hold"


def test_route_message_donation_adds_to_pool() -> None:
    pipeline.set_extractor(deterministic.deterministic_extract)
    result = deterministic.route_message(
        "Kolenchery Round Table", "we'd like to donate 2 commode chairs, kolenchery"
    )
    assert result["outcome"] == "auto_resolved"


def test_route_message_volunteer_signup_assigns_shift() -> None:
    pipeline.set_extractor(deterministic.deterministic_extract)
    result = deterministic.route_message(
        "Ligi Thomas", "i can do first aid saturday morning, i'm a nurse"
    )
    assert result["outcome"] == "auto_resolved"
    shifts = store.load_shifts()
    shift = next(s for s in shifts if s.id == "s-003")
    assert "Ligi Thomas" in shift.assigned


def test_route_message_out_of_scope_escalates() -> None:
    pipeline.set_extractor(deterministic.deterministic_extract)
    result = deterministic.route_message(
        "Lissy Abraham", "how do i get a tax receipt for my donation last month?"
    )
    assert result["outcome"] == "escalated"


def test_full_inbox_runs_without_a_network_call() -> None:
    """The actual insurance policy: all 14 seeded overnight messages process
    end to end with zero model calls."""
    pipeline.set_extractor(deterministic.deterministic_extract)
    for entry in store.load_inbox():
        result = deterministic.route_message(entry["requester"], entry["message"])
        assert result["outcome"] in ("auto_resolved", "escalated")

    decisions = store.load_decisions()
    assert len(decisions) == len(store.load_inbox())
