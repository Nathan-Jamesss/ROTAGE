"""Pipeline — wires extraction, matching, threshold and the ledger into the
four operations Quartermaster performs. Plain functions, independently
testable, with the model call injected so the safety-critical paths can be
exercised without a network.

tools.py wraps these as Strands @tool entry points; nothing agent-specific
lives here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import ledger, matching, store, threshold
from .extract import extract_with_rotation
from .schema import (
    Decision,
    ExtractedField,
    Extraction,
    IntakeKind,
    Outcome,
    Request,
    Resource,
    Tier,
)

Extractor = "callable[[str], Extraction]"

# Tools call process_need/process_donation with no explicit extractor, so
# deterministic mode needs a way to redirect them without changing every call
# site. Tests that care about determinism pass `extractor=` explicitly
# instead, which always wins over this override.
_ACTIVE_EXTRACTOR = None


def set_extractor(fn) -> None:
    global _ACTIVE_EXTRACTOR
    _ACTIVE_EXTRACTOR = fn


def reset_extractor() -> None:
    global _ACTIVE_EXTRACTOR
    _ACTIVE_EXTRACTOR = None


def _resolve_extractor(explicit):
    return explicit or _ACTIVE_EXTRACTOR or extract_with_rotation

SKILL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "first_aid": ("first aid", "nurse", "nursing", "medical training", "cpr"),
    "logistics": ("logistics", "loading", "coordination"),
    "driving": ("drive", "driving", "own vehicle", "car"),
    "registration": ("registration", "front desk", "check-in", "check in"),
    "photography": ("photography", "camera", "photos"),
    "translation": ("translation", "translate", "interpreter"),
    "teaching": ("teaching", "tutor", "teacher"),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _backfill_known_name(extraction: Extraction, known_name: str) -> Extraction:
    """The channel always knows who sent a message — a WhatsApp sender, a
    named donor — independent of whether the message text states it. Asking
    someone to restate their own name because the model, correctly, wouldn't
    invent one from the message body would be a needless escalation."""
    if not extraction.requester_name.present and known_name:
        extraction.requester_name = ExtractedField(
            value=known_name, confidence=1.0, source="form"
        )
    return extraction


def process_need(requester: str, message: str, *, extractor=None) -> Decision:
    """A need comes in. Extract, rank against the donation pool, decide."""
    extraction = _resolve_extractor(extractor)(message)
    extraction.kind = IntakeKind.NEED
    extraction = threshold.merge_vulnerable_flags(extraction, message)
    extraction = _backfill_known_name(extraction, requester)

    request = Request(
        id=store.next_id("q", store.load_requests()),
        requester=requester,
        raw_message=message,
        extraction=extraction,
        logged_at=_now(),
    )
    store.add_request(request)

    pool = store.load_resources()
    candidates = matching.rank(extraction, pool)
    candidates = threshold.filter_auto_matchable(extraction, candidates)

    outcome, tier, rules, reason = threshold.evaluate(extraction, candidates)
    confidence = threshold.confidence_of(extraction, candidates)

    # An escalated request stays "open" rather than "escalated": whether a
    # coordinator needs to look at it lives in the ledger and the triage
    # queue, not here. Closing it out of the pool would mean a donation that
    # arrives later — often the very thing that resolves the escalation —
    # could never find it. The one case that must NOT auto-link later is a
    # vulnerable request, and that is guarded at donation-match time by
    # filter_auto_matchable_reverse, not by taking the request out of the pool.
    chosen = None
    if outcome == Outcome.AUTO_RESOLVED and candidates:
        chosen = candidates[0].resource_id
        store.reserve_resource(chosen, request.id)
        request.status = "matched"
    store.update_request(request)

    return ledger.create(
        request_id=request.id,
        kind=IntakeKind.NEED,
        raw_message=message,
        requester=requester,
        extraction=extraction,
        candidates=candidates,
        chosen=chosen,
        rules=rules,
        outcome=outcome,
        tier=tier,
        confidence=confidence,
        escalation_reason=reason,
    )


def process_donation(donor: str, message: str, *, extractor=None) -> Decision:
    """A donation comes in. Add it to the pool, then search open needs in
    reverse — the same matching engine, pointed the other way."""
    extraction = _resolve_extractor(extractor)(message)
    extraction.kind = IntakeKind.DONATION
    extraction = _backfill_known_name(extraction, donor)

    try:
        quantity = int(float(extraction.quantity.value))
    except (TypeError, ValueError):
        quantity = 1

    resource = Resource(
        id=store.next_id("r", store.load_resources()),
        donor=donor,
        category=extraction.category.value or "other",
        item=str(extraction.item.value or extraction.category.value or "donation"),
        quantity=quantity,
        location=str(extraction.location.value or ""),
        logged_at=_now(),
    )
    store.add_resource(resource)

    open_needs = [r for r in store.load_requests() if r.status == "open"]
    candidates = matching.rank_reverse(extraction, open_needs)
    need_extractions = {r.id: r.extraction for r in open_needs}
    candidates = threshold.filter_auto_matchable_reverse(candidates, need_extractions)

    outcome, tier, rules, reason = threshold.evaluate(extraction, candidates)
    confidence = threshold.confidence_of(extraction, candidates)

    # A donation with no plausible open need is not a problem to escalate —
    # it just sits in the pool waiting for a need to arrive.
    if outcome == Outcome.ESCALATED and tier == Tier.UNMATCHED:
        was_withheld = any(c.blocked for c in candidates)
        if was_withheld:
            tier = Tier.SAFETY_HOLD
            reason = (
                "A matching need exists but involves a vulnerable person. "
                "Withheld from automatic linking."
            )
        else:
            outcome, tier, reason = Outcome.AUTO_RESOLVED, Tier.AUTO_RESOLVED, None
            rules = rules + [
                threshold.RuleResult(
                    rule="no_open_need",
                    passed=True,
                    statement="added to pool, no matching need yet",
                )
            ]

    chosen = None
    if outcome == Outcome.AUTO_RESOLVED and candidates and candidates[0].score >= threshold.MIN_MATCH_SCORE:
        matched_request_id = candidates[0].resource_id
        store.reserve_resource(resource.id, matched_request_id)
        requests = store.load_requests()
        matched = next((r for r in requests if r.id == matched_request_id), None)
        if matched:
            matched.status = "matched"
            store.update_request(matched)
        chosen = resource.id

    return ledger.create(
        request_id=resource.id,
        kind=IntakeKind.DONATION,
        raw_message=message,
        requester=donor,
        extraction=extraction,
        candidates=candidates,
        chosen=chosen,
        rules=rules,
        outcome=outcome,
        tier=tier,
        confidence=confidence,
        escalation_reason=reason,
    )


def _infer_skill_match(message: str, required_skill: str | None) -> bool:
    if not required_skill:
        return True
    text = message.lower()
    keywords = SKILL_SYNONYMS.get(required_skill, (required_skill.replace("_", " "),))
    return any(keyword in text for keyword in keywords)


def process_shift_signup(volunteer_name: str, message: str, slot_id: str) -> Decision:
    """A volunteer offers time for a shift. Same match-or-escalate pattern,
    a different resource type: a slot instead of a good."""
    shifts = store.load_shifts()
    shift = next((s for s in shifts if s.id == slot_id), None)
    if shift is None:
        raise KeyError(f"no shift with id {slot_id}")

    already_booked = volunteer_name in shift.assigned
    known = next(
        (v for v in store.load_volunteers() if v.name.lower() == volunteer_name.lower()),
        None,
    )
    if known and shift.required_skill:
        has_skill = shift.required_skill in known.skills
    else:
        has_skill = _infer_skill_match(message, shift.required_skill)

    outcome, tier, rules, reason = threshold.evaluate_shift(
        shift_full=shift.full,
        already_booked=already_booked,
        has_required_skill=has_skill,
        required_skill=shift.required_skill,
    )
    confidence = 0.9 if outcome == Outcome.AUTO_RESOLVED else 0.6

    chosen = None
    if outcome == Outcome.AUTO_RESOLVED:
        shift.assigned.append(volunteer_name)
        store.save_shifts(shifts)
        chosen = shift.id

    return ledger.create(
        request_id=slot_id,
        kind=IntakeKind.VOLUNTEER,
        raw_message=message,
        requester=volunteer_name,
        extraction=Extraction(kind=IntakeKind.VOLUNTEER),
        candidates=[],
        chosen=chosen,
        rules=rules,
        outcome=outcome,
        tier=tier,
        confidence=confidence,
        escalation_reason=reason,
    )
