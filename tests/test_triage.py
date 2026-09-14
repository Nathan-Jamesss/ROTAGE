"""Triage ordering and queue summary."""

from __future__ import annotations

from quartermaster import triage
from quartermaster.schema import Decision, HumanAction, Outcome, Tier


def _decision(tier: Tier, confidence: float, human_action=None, **overrides) -> Decision:
    defaults = dict(
        id=f"d-{tier.value}-{confidence}",
        request_id="q-001",
        outcome=Outcome.ESCALATED if tier != Tier.AUTO_RESOLVED else Outcome.AUTO_RESOLVED,
        tier=tier,
        confidence=confidence,
        human_action=human_action,
    )
    defaults.update(overrides)
    return Decision(**defaults)


def test_tier_order_is_worst_first() -> None:
    decisions = [
        _decision(Tier.AUTO_RESOLVED, 0.9),
        _decision(Tier.UNMATCHED, 0.5),
        _decision(Tier.SAFETY_HOLD, 0.9),
        _decision(Tier.CONTESTED, 0.7),
    ]
    ordered = triage.order(decisions)
    assert [d.tier for d in ordered] == [
        Tier.SAFETY_HOLD,
        Tier.CONTESTED,
        Tier.UNMATCHED,
        Tier.AUTO_RESOLVED,
    ]


def test_ties_break_on_ascending_confidence() -> None:
    shaky = _decision(Tier.URGENT_INCOMPLETE, 0.3, id="d-shaky")
    solid = _decision(Tier.URGENT_INCOMPLETE, 0.65, id="d-solid")
    ordered = triage.order([solid, shaky])
    assert ordered[0].id == "d-shaky"


def test_needs_human_excludes_resolved_tiers() -> None:
    decisions = [
        _decision(Tier.SAFETY_HOLD, 0.9, id="d-1"),
        _decision(Tier.AUTO_RESOLVED, 0.9, id="d-2"),
        _decision(Tier.OUT_OF_SCOPE, 0.9, id="d-3"),
    ]
    pending = triage.needs_human(decisions)
    assert [d.id for d in pending] == ["d-1"]


def test_needs_human_excludes_items_a_human_already_actioned() -> None:
    decisions = [
        _decision(Tier.CONTESTED, 0.7, human_action=HumanAction.ACCEPT, id="d-1"),
        _decision(Tier.CONTESTED, 0.6, id="d-2"),
    ]
    pending = triage.needs_human(decisions)
    assert [d.id for d in pending] == ["d-2"]


def test_handled_is_the_complement_of_needs_human() -> None:
    decisions = [
        _decision(Tier.SAFETY_HOLD, 0.9, id="d-1"),
        _decision(Tier.AUTO_RESOLVED, 0.9, id="d-2"),
    ]
    assert {d.id for d in triage.needs_human(decisions)} | {
        d.id for d in triage.handled(decisions)
    } == {"d-1", "d-2"}


def test_empty_queue_renders_without_error() -> None:
    summary = triage.summarize([])
    assert summary.total == 0
    assert summary.needs_you == 0
    assert "0 arrived overnight" in summary.headline()


def test_summary_counts_match_the_daybreak_headline() -> None:
    decisions = [
        _decision(Tier.AUTO_RESOLVED, 0.9, id=f"d-auto-{i}") for i in range(11)
    ] + [
        _decision(Tier.SAFETY_HOLD, 0.9, id="d-1"),
        _decision(Tier.CONTESTED, 0.7, id="d-2"),
        _decision(Tier.UNMATCHED, 0.6, id="d-3"),
    ]
    summary = triage.summarize(decisions)
    assert summary.total == 14
    assert summary.handled == 11
    assert summary.needs_you == 3
    assert summary.headline() == "14 arrived overnight    11 handled    3 need you"


def test_by_tier_groups_every_tier_even_when_empty() -> None:
    grouped = triage.by_tier([_decision(Tier.SAFETY_HOLD, 0.9, id="d-1")])
    assert set(grouped.keys()) == set(Tier)
    assert len(grouped[Tier.SAFETY_HOLD]) == 1
    assert grouped[Tier.CONTESTED] == []
