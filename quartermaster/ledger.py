"""Ledger — the immutable record of everything Quartermaster did, and the
receipt format a coordinator actually reads.

Nothing here is ever deleted. Undo reverses an effect (a reservation, an
assignment) and appends its own record; it does not erase the original.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import store
from .schema import (
    Candidate,
    Decision,
    Extraction,
    HumanAction,
    IntakeKind,
    Outcome,
    RuleResult,
    Tier,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create(
    *,
    request_id: str,
    kind: IntakeKind,
    raw_message: str,
    requester: str,
    extraction: Extraction,
    candidates: list[Candidate],
    chosen: str | None,
    rules: list[RuleResult],
    outcome: Outcome,
    tier: Tier,
    confidence: float,
    escalation_reason: str | None,
) -> Decision:
    """Build and persist one decision record."""
    decision = Decision(
        id=store.next_id("d", store.load_decisions()),
        request_id=request_id,
        created_at=_now(),
        kind=kind,
        raw_message=raw_message,
        requester=requester,
        extraction=extraction,
        candidates=candidates,
        chosen=chosen,
        rules=rules,
        outcome=outcome,
        tier=tier,
        confidence=confidence,
        escalation_reason=escalation_reason,
        reversible=(outcome == Outcome.AUTO_RESOLVED and chosen is not None),
    )
    store.add_decision(decision)
    return decision


def apply_human_action(
    decision_id: str, action: HumanAction, note: str | None = None
) -> Decision:
    """Record what a coordinator did with an escalated item.

    accept / edit / respond / ignore — Agent Inbox's vocabulary. This does not
    itself change resource reservations; a coordinator's edit or accept on a
    contested match is applied by the caller before this is recorded.
    """
    decision = store.get_decision(decision_id)
    if decision is None:
        raise KeyError(f"no decision with id {decision_id}")
    decision.human_action = action
    decision.human_note = note
    store.update_decision(decision)
    return decision


def undo(decision_id: str) -> Decision:
    """Reverse an auto-resolution. Releases the reservation, keeps the record.

    The original decision is never deleted or rewritten to look like it didn't
    happen; it is marked reversed, with a timestamp, alongside everything it
    originally said.
    """
    decision = store.get_decision(decision_id)
    if decision is None:
        raise KeyError(f"no decision with id {decision_id}")
    if not decision.reversible:
        raise ValueError(f"decision {decision_id} is not reversible")
    if decision.outcome == Outcome.REVERSED:
        raise ValueError(f"decision {decision_id} was already reversed")

    if decision.chosen:
        try:
            store.release_resource(decision.chosen)
        except KeyError:
            pass  # shift assignments are released by the caller instead

    decision.outcome = Outcome.REVERSED
    decision.reversed_at = _now()
    store.update_decision(decision)
    return decision


def render_receipt(decision: Decision) -> str:
    """Render a decision the way a coordinator reads it: what came in, what
    was extracted, which rules fired, and why it landed where it landed."""
    lines: list[str] = []
    ts = decision.created_at.strftime("%H:%M") if decision.created_at else "--:--"
    lines.append(f"DECISION {decision.id} · {ts}")
    lines.append("─" * 56)
    lines.append(f'"{decision.raw_message}"')
    lines.append(f"                              — {decision.requester}")
    lines.append("")
    lines.append("EXTRACTED")
    for name in ("category", "item", "quantity", "location"):
        field = decision.extraction.field(name)
        if field.present:
            lines.append(f"  {name:<16}  {field.value!s:<20} {field.confidence:.2f}")
    if decision.extraction.vulnerable_flags:
        flags = ", ".join(f.value for f in decision.extraction.vulnerable_flags)
        lines.append(f"  vulnerable_flags  [{flags}]")
        if decision.extraction.vulnerable_evidence:
            lines.append(f'    evidence: "{decision.extraction.vulnerable_evidence}"')
    lines.append("")
    lines.append("RULES EVALUATED")
    for rule in decision.rules:
        mark = "✓" if rule.passed else "✗"
        lines.append(f"  {mark} {rule.rule:<24} {rule.statement}")
    lines.append("")
    lines.append(f"OUTCOME    {decision.outcome.value.upper()} · Tier {decision.tier.value}")
    if decision.escalation_reason:
        lines.append(f"REASON     {decision.escalation_reason}")
    if decision.human_action:
        lines.append(f"HUMAN      {decision.human_action.value}"
                      + (f" — {decision.human_note}" if decision.human_note else ""))
    if decision.reversed_at:
        lines.append(f"REVERSED   {decision.reversed_at.strftime('%H:%M')}")
    return "\n".join(lines)
