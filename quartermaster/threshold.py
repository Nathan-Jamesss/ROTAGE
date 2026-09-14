"""Threshold — the line Quartermaster will not cross on its own.

Pure functions. No LLM call, no I/O, no network. Everything here is
deterministic and testable, which is the point: the safety guarantee is
something you can run, not something the system prompt promises.

The central rule of this module is structural. Candidates that must never be
auto-assigned are removed from the list *before* the agent sees them. The model
is not asked to show restraint; it is never offered the option. That cannot be
undone by a clever prompt, a jailbreak, or a bad sampling run.
"""

from __future__ import annotations

import re

from .schema import (
    Candidate,
    Extraction,
    Outcome,
    RuleResult,
    Tier,
    VulnerableFlag,
)

MIN_MATCH_SCORE = 75.0
TIE_MARGIN = 10.0
MIN_FIELD_CONFIDENCE = 0.70
MAX_AUTO_QUANTITY = 20
REQUIRED_FIELDS = ("category", "quantity", "requester_name")


VULNERABLE_KEYWORDS: dict[VulnerableFlag, tuple[str, ...]] = {
    VulnerableFlag.MINOR: (
        "child", "children", "kid", "kids", "baby", "babies", "infant",
        "toddler", "son", "daughter", "minor", "students", "schoolchildren",
        "children's home", "orphan", "newborn",
    ),
    VulnerableFlag.ELDERLY: (
        "elderly", "grandmother", "grandfather", "granny", "grandma",
        "grandpa", "senior citizen", "old age", "aged parent", "paati",
        "thatha",
    ),
    VulnerableFlag.MEDICAL: (
        "medical", "medicine", "medicines", "hospital", "surgery", "ill",
        "unwell", "sick", "diabetic", "heart condition", "treatment",
        "cancer", "dialysis", "injured", "fever", "not well", "bedridden",
    ),
    VulnerableFlag.DISABILITY: (
        "disabled", "disability", "blind", "deaf", "special needs",
        "wheelchair", "differently abled",
    ),
    VulnerableFlag.SAFETY: (
        "unsafe", "violence", "abuse", "eviction", "evicted", "homeless",
        "shelter", "threatened", "no roof",
    ),
    VulnerableFlag.PREGNANCY: ("pregnant", "pregnancy", "expecting"),
}

_AGE_PATTERN = re.compile(r"\b(\d{1,3})\s*(?:years?\s*old|yrs?|yo)\b", re.I)


def detect_vulnerable_flags(text: str) -> tuple[list[VulnerableFlag], str | None]:
    """Deterministic keyword scan, run in addition to the model's extraction.

    Belt and braces. The safety path never depends on the model alone, so this
    result is unioned with whatever the model reported, never intersected.
    """
    if not text:
        return [], None

    lowered = text.lower()
    found: list[VulnerableFlag] = []
    evidence: list[str] = []

    for flag, keywords in VULNERABLE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                if flag not in found:
                    found.append(flag)
                evidence.append(keyword)
                break

    for match in _AGE_PATTERN.finditer(text):
        age = int(match.group(1))
        if age >= 65 and VulnerableFlag.ELDERLY not in found:
            found.append(VulnerableFlag.ELDERLY)
            evidence.append(match.group(0))
        elif age < 18 and VulnerableFlag.MINOR not in found:
            found.append(VulnerableFlag.MINOR)
            evidence.append(match.group(0))

    return found, ", ".join(evidence) if evidence else None


def merge_vulnerable_flags(extraction: Extraction, raw_text: str) -> Extraction:
    """Union the model's flags with the keyword scan. Either one is enough."""
    scanned, evidence = detect_vulnerable_flags(raw_text)
    combined = list(extraction.vulnerable_flags)
    for flag in scanned:
        if flag not in combined:
            combined.append(flag)

    extraction.vulnerable_flags = combined
    if evidence and not extraction.vulnerable_evidence:
        extraction.vulnerable_evidence = evidence
    return extraction


def filter_auto_matchable(
    extraction: Extraction, candidates: list[Candidate]
) -> list[Candidate]:
    """Remove candidates that must never be auto-assigned.

    Called before candidates are returned to the agent. A blocked case comes
    back with every candidate marked, so the agent can see that options existed
    and were withheld, but cannot act on any of them.
    """
    if not extraction.vulnerable_flags:
        return candidates

    flags = ", ".join(f.value for f in extraction.vulnerable_flags)
    reason = f"T1 vulnerable_person [{flags}] — withheld from automatic assignment"
    return [
        candidate.model_copy(update={"blocked": True, "blocked_reason": reason})
        for candidate in candidates
    ]


def _quantity_of(extraction: Extraction) -> int | None:
    raw = extraction.quantity.value
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def evaluate(
    extraction: Extraction, candidates: list[Candidate]
) -> tuple[Outcome, Tier, list[RuleResult], str | None]:
    """Decide what happens to one intake. Pure and deterministic.

    Returns the outcome, its triage tier, every rule evaluated in order, and a
    human-readable escalation reason when one applies.
    """
    rules: list[RuleResult] = []
    usable = [c for c in candidates if not c.blocked]
    usable.sort(key=lambda c: c.score, reverse=True)
    top = usable[0] if usable else None
    second = usable[1] if len(usable) > 1 else None

    # T1 — vulnerable person. Checked first and short-circuits: no candidate
    # search result can override it, and no later rule can rescue it.
    if extraction.vulnerable_flags:
        flags = ", ".join(f.value for f in extraction.vulnerable_flags)
        rules.append(
            RuleResult(
                rule="T1_vulnerable_person",
                passed=False,
                statement=f"flags=[{flags}] -> automatic assignment blocked",
            )
        )
        reason = (
            f"Mentions a vulnerable person ({flags}). Quartermaster does not "
            f"auto-assign resources in these cases. A coordinator decides."
        )
        return Outcome.ESCALATED, Tier.SAFETY_HOLD, rules, reason
    rules.append(
        RuleResult(
            rule="T1_vulnerable_person",
            passed=True,
            statement="no vulnerable flags detected",
        )
    )

    # T6 — required fields present at all.
    missing = [
        name for name in REQUIRED_FIELDS if not extraction.field(name).present
    ]
    if missing:
        rules.append(
            RuleResult(
                rule="T6_missing_required_field",
                passed=False,
                statement=f"missing={missing}",
            )
        )
        question = (
            extraction.followup_questions[0]
            if extraction.followup_questions
            else f"Ask the requester for: {', '.join(missing)}"
        )
        return (
            Outcome.ESCALATED,
            Tier.URGENT_INCOMPLETE,
            rules,
            f"Required information missing ({', '.join(missing)}). {question}",
        )
    rules.append(
        RuleResult(
            rule="T6_missing_required_field",
            passed=True,
            statement=f"all present: {', '.join(REQUIRED_FIELDS)}",
        )
    )

    # T5 — present but too shaky to trust. Treated the same as missing.
    shaky = [
        f"{name}({extraction.field(name).confidence:.2f})"
        for name in REQUIRED_FIELDS
        if extraction.field(name).confidence < MIN_FIELD_CONFIDENCE
    ]
    if shaky:
        rules.append(
            RuleResult(
                rule="T5_low_confidence_field",
                passed=False,
                statement=f"{', '.join(shaky)} < {MIN_FIELD_CONFIDENCE}",
            )
        )
        return (
            Outcome.ESCALATED,
            Tier.URGENT_INCOMPLETE,
            rules,
            f"Extraction confidence too low on {', '.join(shaky)}. "
            f"Worth a human reading the original message.",
        )
    rules.append(
        RuleResult(
            rule="T5_low_confidence_field",
            passed=True,
            statement=f"all required fields >= {MIN_FIELD_CONFIDENCE}",
        )
    )

    # T7 — the message disagrees with itself. Never averaged, never guessed.
    if extraction.contradictions:
        detail = "; ".join(extraction.contradictions)
        rules.append(
            RuleResult(
                rule="T7_contradiction", passed=False, statement=detail
            )
        )
        return (
            Outcome.ESCALATED,
            Tier.CONTESTED,
            rules,
            f"The message contradicts itself ({detail}). Quartermaster will not "
            f"pick a side.",
        )
    rules.append(
        RuleResult(
            rule="T7_contradiction", passed=True, statement="no contradictions found"
        )
    )

    # T2 — unusually large ask.
    quantity = _quantity_of(extraction)
    if quantity is not None and quantity > MAX_AUTO_QUANTITY:
        rules.append(
            RuleResult(
                rule="T2_oversized_request",
                passed=False,
                statement=f"quantity {quantity} > {MAX_AUTO_QUANTITY}",
            )
        )
        return (
            Outcome.ESCALATED,
            Tier.CONTESTED,
            rules,
            f"Unusually large ({quantity} units). Worth a coordinator's eyes "
            f"before committing the pool.",
        )
    rules.append(
        RuleResult(
            rule="T2_oversized_request",
            passed=True,
            statement=f"quantity {quantity} <= {MAX_AUTO_QUANTITY}",
        )
    )

    # T4 — nothing in the pool is good enough.
    if top is None or top.score < MIN_MATCH_SCORE:
        best = f"{top.score:.1f}" if top else "none"
        rules.append(
            RuleResult(
                rule="T4_no_viable_match",
                passed=False,
                statement=f"best score {best} < {MIN_MATCH_SCORE}",
            )
        )
        return (
            Outcome.ESCALATED,
            Tier.UNMATCHED,
            rules,
            "Nothing in the pool fits this. Needs sourcing.",
        )
    rules.append(
        RuleResult(
            rule="T4_no_viable_match",
            passed=True,
            statement=f"best score {top.score:.1f} >= {MIN_MATCH_SCORE}",
        )
    )

    # T3 — two candidates too close to separate.
    if second is not None and (top.score - second.score) < TIE_MARGIN:
        margin = top.score - second.score
        rules.append(
            RuleResult(
                rule="T3_contested_match",
                passed=False,
                statement=(
                    f"{top.resource_id} {top.score:.1f} vs "
                    f"{second.resource_id} {second.score:.1f}, "
                    f"margin {margin:.1f} < {TIE_MARGIN}"
                ),
            )
        )
        return (
            Outcome.ESCALATED,
            Tier.CONTESTED,
            rules,
            f"Two close candidates ({top.resource_id} and {second.resource_id}). "
            f"A coin flip is not a decision.",
        )
    margin_text = (
        f"margin {top.score - second.score:.1f} >= {TIE_MARGIN}"
        if second is not None
        else "single candidate, uncontested"
    )
    rules.append(
        RuleResult(rule="T3_contested_match", passed=True, statement=margin_text)
    )

    return Outcome.AUTO_RESOLVED, Tier.AUTO_RESOLVED, rules, None


def confidence_of(extraction: Extraction, candidates: list[Candidate]) -> float:
    """Overall confidence in a decision: the weakest required field, tempered
    by how good the best match was. Deliberately pessimistic."""
    field_scores = [extraction.field(n).confidence for n in REQUIRED_FIELDS]
    weakest = min(field_scores) if field_scores else 0.0

    usable = [c for c in candidates if not c.blocked]
    if not usable:
        return round(weakest, 2)

    best = max(c.score for c in usable) / 100.0
    return round(min(weakest, best), 2)


def evaluate_shift(
    shift_full: bool,
    already_booked: bool,
    has_required_skill: bool,
    required_skill: str | None,
) -> tuple[Outcome, Tier, list[RuleResult], str | None]:
    """Shift assignment runs the same match-or-escalate pattern against a
    different resource type: time instead of goods."""
    rules: list[RuleResult] = []

    if shift_full:
        rules.append(
            RuleResult(
                rule="T8_shift_conflict",
                passed=False,
                statement="slot already at capacity",
            )
        )
        return (
            Outcome.ESCALATED,
            Tier.CONTESTED,
            rules,
            "That slot is already full. Someone has to decide who takes it.",
        )
    rules.append(
        RuleResult(rule="T8_shift_conflict", passed=True, statement="slot has room")
    )

    if already_booked:
        rules.append(
            RuleResult(
                rule="T8_double_booking",
                passed=False,
                statement="volunteer already assigned to this slot",
            )
        )
        return (
            Outcome.ESCALATED,
            Tier.CONTESTED,
            rules,
            "This volunteer is already booked for that slot.",
        )
    rules.append(
        RuleResult(
            rule="T8_double_booking", passed=True, statement="no double booking"
        )
    )

    if not has_required_skill:
        rules.append(
            RuleResult(
                rule="T9_skill_mismatch",
                passed=False,
                statement=f"slot requires '{required_skill}'",
            )
        )
        return (
            Outcome.ESCALATED,
            Tier.CONTESTED,
            rules,
            f"This role needs '{required_skill}' and the volunteer has not "
            f"listed it. A coordinator may still know they can do it.",
        )
    rules.append(
        RuleResult(
            rule="T9_skill_mismatch",
            passed=True,
            statement=f"skill requirement satisfied ({required_skill or 'none'})",
        )
    )

    return Outcome.AUTO_RESOLVED, Tier.AUTO_RESOLVED, rules, None
