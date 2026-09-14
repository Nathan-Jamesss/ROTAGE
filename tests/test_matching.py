"""Matching tests. A score a coordinator cannot interrogate is a score they
will not trust, so the derivation is tested as carefully as the result."""

from __future__ import annotations

from quartermaster import matching
from quartermaster.schema import (
    Category,
    ExtractedField,
    Extraction,
    Request,
    Resource,
)


def _resource(**overrides) -> Resource:
    defaults = dict(
        id="r-001",
        donor="Sundaram Textiles",
        category=Category.BEDDING,
        item="12 woollen blankets, new",
        quantity=12,
        unit="pieces",
        location="Ward 7",
    )
    defaults.update(overrides)
    return Resource(**defaults)


def _need(**overrides) -> Extraction:
    base = Extraction(
        category=ExtractedField(value="bedding", confidence=0.94),
        item=ExtractedField(value="blankets", confidence=0.9),
        quantity=ExtractedField(value=4, confidence=0.88),
        location=ExtractedField(value="Ward 7", confidence=0.91),
        requester_name=ExtractedField(value="Meena R.", confidence=0.97),
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_exact_category_beats_adjacent_beats_unrelated() -> None:
    need = _need()
    exact = matching.rank(need, [_resource()])[0]
    adjacent = matching.rank(
        need, [_resource(category=Category.CLOTHING, item="warm shawls")]
    )[0]
    unrelated = matching.rank(
        need, [_resource(category=Category.TRANSPORT, item="van trips")]
    )[0]

    assert exact.score > adjacent.score > unrelated.score


def test_fuzzy_item_match_survives_extra_words() -> None:
    """"blankets" has to find "12 woollen blankets, new"."""
    candidate = matching.rank(_need(), [_resource()])[0]
    assert candidate.breakdown["item_similarity"] == matching.W_ITEM


def test_same_category_different_wording_still_matches() -> None:
    """"school supplies" and "40 notebook sets" are the same thing.

    Without a category-backed floor this scores too low to ever auto-resolve,
    which would make the agent useless on exactly the requests it should handle.
    """
    need = _need(
        category=ExtractedField(value="school_supplies", confidence=0.93),
        item=ExtractedField(value="school supplies", confidence=0.9),
        quantity=ExtractedField(value=4, confidence=0.9),
    )
    resource = _resource(
        category=Category.SCHOOL_SUPPLIES,
        item="40 notebook sets with geometry boxes",
        quantity=40,
    )
    candidate = matching.rank(need, [resource])[0]
    assert candidate.score >= 75.0


def test_noise_floor_gives_no_credit_to_unrelated_wording() -> None:
    need = _need()
    candidate = matching.rank(
        need, [_resource(category=Category.TRANSPORT, item="van trips to hospital")]
    )[0]
    assert candidate.breakdown["item_similarity"] == 0.0


def test_insufficient_quantity_reduces_score() -> None:
    plenty = matching.rank(_need(), [_resource(quantity=12)])[0]
    scarce = matching.rank(_need(), [_resource(quantity=1)])[0]

    assert plenty.breakdown["qty_sufficiency"] > scarce.breakdown["qty_sufficiency"]
    assert plenty.score > scarce.score


def test_same_area_beats_different_area() -> None:
    near = matching.rank(_need(), [_resource(location="Ward 7")])[0]
    far = matching.rank(_need(), [_resource(location="Ward 22")])[0]
    assert near.score > far.score


def test_breakdown_sums_to_score() -> None:
    candidate = matching.rank(_need(), [_resource()])[0]
    assert abs(sum(candidate.breakdown.values()) - candidate.score) < 0.15


def test_every_component_is_explained() -> None:
    candidate = matching.rank(_need(), [_resource()])[0]
    assert len(candidate.explanations) == len(candidate.breakdown)
    assert all(text.strip() for text in candidate.explanations)


def test_results_are_ranked_best_first() -> None:
    pool = [
        _resource(id="r-001", location="Ward 22", category=Category.TRANSPORT),
        _resource(id="r-002", location="Ward 7"),
        _resource(id="r-003", location="Ward 22", category=Category.CLOTHING),
    ]
    ranked = matching.rank(_need(), pool)
    assert ranked[0].resource_id == "r-002"
    assert [c.score for c in ranked] == sorted(
        (c.score for c in ranked), reverse=True
    )


def test_reserved_resources_are_excluded() -> None:
    pool = [_resource(reserved_for="someone else")]
    assert matching.rank(_need(), pool) == []


def test_donation_searches_open_needs_with_the_same_engine() -> None:
    """The bidirectional claim, tested. A donation finds the need that wanted it."""
    donation = Extraction(
        category=ExtractedField(value="bedding", confidence=0.95),
        item=ExtractedField(value="woollen blankets", confidence=0.92),
        quantity=ExtractedField(value=12, confidence=0.99),
        location=ExtractedField(value="Ward 7", confidence=0.9),
        requester_name=ExtractedField(value="Sundaram Textiles", confidence=0.99),
    )
    needs = [
        Request(
            id="q-001",
            requester="Meena R.",
            raw_message="need blankets",
            extraction=_need(),
        ),
        Request(
            id="q-002",
            requester="Arun K.",
            raw_message="need a van",
            extraction=_need(
                category=ExtractedField(value="transport", confidence=0.9),
                item=ExtractedField(value="van for hospital trip", confidence=0.9),
            ),
        ),
    ]

    ranked = matching.rank_reverse(donation, needs)
    assert ranked[0].resource_id == "q-001"
    assert ranked[0].score > ranked[1].score


def test_closed_needs_are_ignored_in_reverse() -> None:
    donation = _need()
    needs = [
        Request(
            id="q-001",
            requester="Meena R.",
            raw_message="need blankets",
            extraction=_need(),
            status="closed",
        )
    ]
    assert matching.rank_reverse(donation, needs) == []


def test_unknown_category_scores_zero_for_that_component() -> None:
    need = _need(category=ExtractedField(value=None, confidence=0.0))
    candidate = matching.rank(need, [_resource()])[0]
    assert candidate.breakdown["category_match"] == 0.0
