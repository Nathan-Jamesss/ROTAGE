"""Triage — turns a pile of decisions into the queue a coordinator reads.

Six tiers, worst-first. Within a tier, ascending confidence: the cases
Quartermaster was least sure about surface before the ones it was fairly
sure about, on the theory that a human's attention is the scarce resource and
the shakiest calls deserve it first.
"""

from __future__ import annotations

from .schema import Decision, Tier

TIER_LABELS: dict[Tier, str] = {
    Tier.SAFETY_HOLD: "Safety hold",
    Tier.URGENT_INCOMPLETE: "Urgent, incomplete",
    Tier.CONTESTED: "Contested",
    Tier.UNMATCHED: "Unmatched",
    Tier.AUTO_RESOLVED: "Auto-resolved",
    Tier.OUT_OF_SCOPE: "Out of scope",
}

NEEDS_HUMAN_TIERS = (
    Tier.SAFETY_HOLD,
    Tier.URGENT_INCOMPLETE,
    Tier.CONTESTED,
    Tier.UNMATCHED,
)


def sort_key(decision: Decision) -> tuple[str, float]:
    # Tier values are "1_safety_hold" .. "6_out_of_scope"; the leading digit
    # makes lexicographic order the same as tier order.
    return (decision.tier.value, decision.confidence)


def order(decisions: list[Decision]) -> list[Decision]:
    return sorted(decisions, key=sort_key)


def needs_human(decisions: list[Decision]) -> list[Decision]:
    return order(
        [
            d
            for d in decisions
            if d.tier in NEEDS_HUMAN_TIERS and d.human_action is None
        ]
    )


def handled(decisions: list[Decision]) -> list[Decision]:
    return order(
        [
            d
            for d in decisions
            if d.tier not in NEEDS_HUMAN_TIERS or d.human_action is not None
        ]
    )


def by_tier(decisions: list[Decision]) -> dict[Tier, list[Decision]]:
    grouped: dict[Tier, list[Decision]] = {tier: [] for tier in Tier}
    for decision in order(decisions):
        grouped[decision.tier].append(decision)
    return grouped


class QueueSummary:
    def __init__(self, decisions: list[Decision]) -> None:
        self.total = len(decisions)
        self.needs_you = len(needs_human(decisions))
        self.handled = self.total - self.needs_you

    def headline(self) -> str:
        return (
            f"{self.total} arrived overnight    "
            f"{self.handled} handled    "
            f"{self.needs_you} need you"
        )


def summarize(decisions: list[Decision]) -> QueueSummary:
    return QueueSummary(decisions)
