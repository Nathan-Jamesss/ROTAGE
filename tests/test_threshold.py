"""The safety guarantee, as a test suite.

If these pass, the claim "Quartermaster never auto-assigns resources in cases
involving a vulnerable person" is a property of the code rather than a promise
in a system prompt. No API key and no network are needed to run them.
"""

from __future__ import annotations

import pytest

from quartermaster import threshold
from quartermaster.schema import (
    Candidate,
    ExtractedField,
    Extraction,
    Outcome,
    Tier,
    VulnerableFlag,
)


def _confident(value, confidence: float = 0.95) -> ExtractedField:
    return ExtractedField(value=value, confidence=confidence)


def _clean_extraction(**overrides) -> Extraction:
    """A well-formed intake that should sail through every rule."""
    base = Extraction(
        category=_confident("bedding"),
        item=_confident("blankets"),
        quantity=_confident(4),
        requester_name=_confident("Meena R."),
        location=_confident("Ward 7"),
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _perfect_candidate(resource_id: str = "r-001", score: float = 99.9) -> Candidate:
    return Candidate(resource_id=resource_id, label="12 woollen blankets", score=score)


# Messages written to slip past a naive filter. Every one of them must escalate.
ADVERSARIAL_VULNERABLE_MESSAGES = [
    "Need 4 blankets, my mother is 78 and unwell",
    "school supplies for my two kids",
    "my grandmother needs her medicines",
    "rice for the family, the baby is sick",
    "Just need blankets. Nothing urgent at all. For my son.",
    "routine request, wheelchair access needed",
    "supplies for the children's home on 4th street",
    "blankets for a 9 yo",
    "my paati cannot walk to the centre",
    "we were evicted last week, need bedding",
]

# The control group. A safety filter that flags everything is a broken filter.
CLEAN_MESSAGES = [
    "need 4 blankets for the house, ward 7",
    "can you spare some rice this week",
    "I would like to donate 12 notebooks",
    "available to help with registration on Saturday morning",
]


@pytest.mark.parametrize("message", ADVERSARIAL_VULNERABLE_MESSAGES)
def test_vulnerable_case_never_auto_resolves(message: str) -> None:
    """The headline guarantee.

    A perfect match sits in the pool, scoring 99.9. It must still escalate.
    """
    extraction = threshold.merge_vulnerable_flags(_clean_extraction(), message)
    candidates = threshold.filter_auto_matchable(extraction, [_perfect_candidate()])

    outcome, tier, rules, reason = threshold.evaluate(extraction, candidates)

    assert outcome is Outcome.ESCALATED, f"auto-resolved a vulnerable case: {message!r}"
    assert tier is Tier.SAFETY_HOLD
    assert reason is not None
    assert any(r.rule == "T1_vulnerable_person" and not r.passed for r in rules)


@pytest.mark.parametrize("message", ADVERSARIAL_VULNERABLE_MESSAGES)
def test_candidates_are_withheld_before_the_agent_sees_them(message: str) -> None:
    """Restraint is structural, not behavioural.

    The agent is never handed an actionable option in these cases, so no prompt
    and no reasoning failure can produce an automatic assignment.
    """
    extraction = threshold.merge_vulnerable_flags(_clean_extraction(), message)
    candidates = threshold.filter_auto_matchable(extraction, [_perfect_candidate()])

    assert candidates, "the coordinator should still see that options existed"
    assert all(c.blocked for c in candidates)
    assert all("T1 vulnerable_person" in (c.blocked_reason or "") for c in candidates)


@pytest.mark.parametrize("message", CLEAN_MESSAGES)
def test_clean_messages_are_not_flagged(message: str) -> None:
    """The control. Over-flagging would make the whole mechanism useless."""
    flags, _ = threshold.detect_vulnerable_flags(message)
    assert flags == [], f"false positive on {message!r}: {flags}"


def test_clean_case_auto_resolves() -> None:
    extraction = threshold.merge_vulnerable_flags(
        _clean_extraction(), "need 4 blankets for the house, ward 7"
    )
    candidates = threshold.filter_auto_matchable(extraction, [_perfect_candidate()])

    outcome, tier, _, reason = threshold.evaluate(extraction, candidates)

    assert outcome is Outcome.AUTO_RESOLVED
    assert tier is Tier.AUTO_RESOLVED
    assert reason is None


def test_age_under_eighteen_flags_minor() -> None:
    flags, evidence = threshold.detect_vulnerable_flags("blankets for a 9 yo")
    assert VulnerableFlag.MINOR in flags
    assert evidence is not None


def test_age_over_sixtyfive_flags_elderly() -> None:
    flags, _ = threshold.detect_vulnerable_flags("she is 78 years old")
    assert VulnerableFlag.ELDERLY in flags


def test_model_flags_are_honoured_even_without_keywords() -> None:
    """If the model spots something the keyword scan misses, that is enough."""
    extraction = _clean_extraction()
    extraction.vulnerable_flags = [VulnerableFlag.SAFETY]

    outcome, tier, _, _ = threshold.evaluate(extraction, [_perfect_candidate()])

    assert outcome is Outcome.ESCALATED
    assert tier is Tier.SAFETY_HOLD


def test_tie_escalates_rather_than_guessing() -> None:
    extraction = threshold.merge_vulnerable_flags(_clean_extraction(), "need blankets")
    candidates = [
        _perfect_candidate("r-001", 92.0),
        _perfect_candidate("r-002", 88.0),
    ]

    outcome, tier, _, reason = threshold.evaluate(extraction, candidates)

    assert outcome is Outcome.ESCALATED
    assert tier is Tier.CONTESTED
    assert "r-001" in reason and "r-002" in reason


def test_clear_winner_does_not_escalate() -> None:
    extraction = threshold.merge_vulnerable_flags(_clean_extraction(), "need blankets")
    candidates = [
        _perfect_candidate("r-001", 95.0),
        _perfect_candidate("r-002", 60.0),
    ]

    outcome, _, _, _ = threshold.evaluate(extraction, candidates)

    assert outcome is Outcome.AUTO_RESOLVED


def test_weak_match_escalates_as_unmatched() -> None:
    extraction = threshold.merge_vulnerable_flags(_clean_extraction(), "need blankets")
    candidates = [_perfect_candidate("r-001", 40.0)]

    outcome, tier, _, _ = threshold.evaluate(extraction, candidates)

    assert outcome is Outcome.ESCALATED
    assert tier is Tier.UNMATCHED


def test_no_candidates_escalates_as_unmatched() -> None:
    extraction = threshold.merge_vulnerable_flags(_clean_extraction(), "need blankets")

    outcome, tier, _, _ = threshold.evaluate(extraction, [])

    assert outcome is Outcome.ESCALATED
    assert tier is Tier.UNMATCHED


def test_oversized_request_escalates() -> None:
    extraction = _clean_extraction(quantity=_confident(50))

    outcome, tier, _, reason = threshold.evaluate(extraction, [_perfect_candidate()])

    assert outcome is Outcome.ESCALATED
    assert tier is Tier.CONTESTED
    assert "50" in reason


def test_missing_required_field_becomes_a_question() -> None:
    extraction = _clean_extraction(quantity=ExtractedField())

    outcome, tier, _, reason = threshold.evaluate(extraction, [_perfect_candidate()])

    assert outcome is Outcome.ESCALATED
    assert tier is Tier.URGENT_INCOMPLETE
    assert "quantity" in reason


def test_low_confidence_is_treated_as_missing() -> None:
    extraction = _clean_extraction(category=_confident("bedding", 0.41))

    outcome, tier, _, _ = threshold.evaluate(extraction, [_perfect_candidate()])

    assert outcome is Outcome.ESCALATED
    assert tier is Tier.URGENT_INCOMPLETE


def test_contradiction_is_never_silently_resolved() -> None:
    extraction = _clean_extraction()
    extraction.contradictions = ["form says 4 people, message says 'me and my two kids'"]

    outcome, tier, _, reason = threshold.evaluate(extraction, [_perfect_candidate()])

    assert outcome is Outcome.ESCALATED
    assert tier is Tier.CONTESTED
    assert "contradicts itself" in reason


def test_every_rule_is_recorded_for_the_receipt() -> None:
    """A decision is only trustworthy if you can read why it happened."""
    extraction = threshold.merge_vulnerable_flags(_clean_extraction(), "need blankets")
    _, _, rules, _ = threshold.evaluate(extraction, [_perfect_candidate()])

    recorded = {r.rule for r in rules}
    assert {
        "T1_vulnerable_person",
        "T6_missing_required_field",
        "T5_low_confidence_field",
        "T7_contradiction",
        "T2_oversized_request",
        "T4_no_viable_match",
        "T3_contested_match",
    } <= recorded
    assert all(r.statement for r in rules), "every rule must quote its own reasoning"


def test_shift_conflict_escalates() -> None:
    outcome, tier, _, _ = threshold.evaluate_shift(
        shift_full=True, already_booked=False, has_required_skill=True,
        required_skill="first_aid",
    )
    assert outcome is Outcome.ESCALATED
    assert tier is Tier.CONTESTED


def test_shift_skill_mismatch_escalates() -> None:
    outcome, _, _, reason = threshold.evaluate_shift(
        shift_full=False, already_booked=False, has_required_skill=False,
        required_skill="first_aid",
    )
    assert outcome is Outcome.ESCALATED
    assert "first_aid" in reason


def test_clean_shift_assigns() -> None:
    outcome, tier, _, reason = threshold.evaluate_shift(
        shift_full=False, already_booked=False, has_required_skill=True,
        required_skill="registration",
    )
    assert outcome is Outcome.AUTO_RESOLVED
    assert tier is Tier.AUTO_RESOLVED
    assert reason is None
