"""Tools — the agent's entire vocabulary.

Each tool is a thin wrapper over pipeline.py. The decision about whether
something can be handled automatically is never made here or by the model
that calls these tools; it was already made, deterministically, inside
threshold.py before this function returns. A tool that comes back with
candidates=[] and a blocked_reason is not a suggestion to be careful — it is
the agent's only view of the situation, and it contains no actionable option.
"""

from __future__ import annotations

from strands import tool

from . import ledger, pipeline, store, triage
from .schema import HumanAction, Tier


def _decision_summary(decision) -> dict:
    return {
        "decision_id": decision.id,
        "outcome": decision.outcome.value,
        "tier": decision.tier.value,
        "confidence": decision.confidence,
        "chosen": decision.chosen,
        "escalation_reason": decision.escalation_reason,
        "candidates": [
            {
                "resource_id": c.resource_id,
                "label": c.label,
                "score": c.score,
                "blocked": c.blocked,
                "blocked_reason": c.blocked_reason,
            }
            for c in decision.candidates
        ],
        "receipt": ledger.render_receipt(decision),
    }


@tool
def log_request(requester: str, message: str) -> dict:
    """Log an incoming need from someone asking for help.

    Extracts what they need, ranks it against the donation pool, and either
    resolves it automatically or escalates it with a specific reason. Use
    this whenever a message is someone asking for equipment, supplies, or
    other help — not for donations or volunteer signups, which have their
    own tools.

    Args:
        requester: The name of the person asking for help.
        message: Their message, in their own words.
    """
    decision = pipeline.process_need(requester, message)
    return _decision_summary(decision)


@tool
def log_donation(donor: str, message: str) -> dict:
    """Log an incoming donation and try to match it to an open need.

    Adds the item to the pool, then searches every currently open need for a
    fit — the same matching logic as log_request, run in the opposite
    direction. Use this when a message is someone offering to give something,
    not asking for it.

    Args:
        donor: The name of the person or club donating.
        message: Their message describing what they're giving.
    """
    decision = pipeline.process_donation(donor, message)
    return _decision_summary(decision)


@tool
def assign_shift(volunteer: str, slot_id: str, message: str) -> dict:
    """Assign a volunteer to an open project shift.

    Checks whether the slot has room, whether this volunteer is already
    booked elsewhere, and whether they have the skill the role requires.
    Escalates instead of guessing on any conflict. Use list_queue or ask a
    coordinator for slot_id if you don't already know which shift a message
    refers to.

    Args:
        volunteer: The volunteer's name.
        slot_id: The shift id, e.g. "s-003".
        message: Their message, used to infer their skill if they are not
            already a known volunteer.
    """
    decision = pipeline.process_shift_signup(volunteer, message, slot_id)
    return _decision_summary(decision)


@tool
def escalate_to_human(reason: str, details: str, request_id: str) -> dict:
    """Escalate something to a coordinator outside the normal intake flow.

    Use this only when you need to raise something that didn't come through
    log_request, log_donation, or assign_shift — those already escalate on
    their own when the rules call for it. This is the fallback for a genuine
    edge case those tools didn't anticipate.

    Args:
        reason: A short machine-readable tag, e.g. "data_contradiction" or
            "needs_sourcing".
        details: A plain explanation a coordinator can act on immediately.
        request_id: The id this escalation relates to, if there is one.
    """
    from .schema import Extraction, IntakeKind, Outcome

    decision = ledger.create(
        request_id=request_id or "manual",
        kind=IntakeKind.NEED,
        raw_message=details,
        requester="(agent-initiated)",
        extraction=Extraction(),
        candidates=[],
        chosen=None,
        rules=[],
        outcome=Outcome.ESCALATED,
        tier=Tier.CONTESTED,
        confidence=0.5,
        escalation_reason=f"[{reason}] {details}",
    )
    return _decision_summary(decision)


@tool
def undo_decision(decision_id: str) -> dict:
    """Reverse an automatic resolution: release the resource or shift slot
    it committed and mark the record reversed. The original decision is kept,
    not deleted.

    Args:
        decision_id: The decision to reverse, e.g. "d-0007".
    """
    decision = ledger.undo(decision_id)
    return _decision_summary(decision)


@tool
def list_queue(tier: str | None = None) -> dict:
    """Return the current coordination queue: what's been handled overnight
    and what still needs a human decision, ordered worst-tier-first.

    Args:
        tier: Optional tier filter, e.g. "1_safety_hold". Omit to see
            everything that still needs a decision.
    """
    decisions = store.load_decisions()
    if tier:
        wanted = Tier(tier)
        pending = [d for d in triage.needs_human(decisions) if d.tier == wanted]
    else:
        pending = triage.needs_human(decisions)

    summary = triage.summarize(decisions)
    return {
        "headline": summary.headline(),
        "total": summary.total,
        "handled": summary.handled,
        "needs_you": summary.needs_you,
        "queue": [_decision_summary(d) for d in pending],
    }


@tool
def apply_human_action(decision_id: str, action: str, note: str = "") -> dict:
    """Record a coordinator's decision on an escalated item.

    Args:
        decision_id: The decision being actioned.
        action: One of "accept", "edit", "respond", "ignore".
        note: An optional note explaining the decision.
    """
    decision = ledger.apply_human_action(
        decision_id, HumanAction(action), note or None
    )
    return _decision_summary(decision)


# --- Stubs. Signatures exist and are documented; not built out for the demo.
# Capabilities #4 and #5 from the project brief need believable time-series
# or RSVP data that isn't worth fabricating under this deadline. Designed to
# extend once real event and follow-up data exists. ---


@tool
def log_rsvp(name: str, event: str, headcount: int) -> dict:
    """STUB. Log an RSVP and headcount for an event. Designed to extend:
    would feed check_event_capacity the running total for that event."""
    return {
        "stub": True,
        "message": (
            f"Recorded {name}'s RSVP for {headcount} at {event!r}. "
            f"Capacity checking is not wired up yet."
        ),
    }


@tool
def check_event_capacity(event: str) -> dict:
    """STUB. Compare an event's RSVP total against venue capacity, escalating
    when overbooked or when projected no-shows would break the plan."""
    return {
        "stub": True,
        "message": f"Capacity tracking for {event!r} is not wired up yet.",
    }


@tool
def log_followup(request_id: str, status: str) -> dict:
    """STUB. Record whether a fulfilled request actually helped, and flag
    cases that need a repeat visit or deeper support than originally logged."""
    return {
        "stub": True,
        "message": (
            f"Recorded follow-up status {status!r} for {request_id}. "
            f"Automatic re-escalation on repeat need is not wired up yet."
        ),
    }
