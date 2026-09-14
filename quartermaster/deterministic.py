"""Deterministic mode — the demo's insurance policy.

Runs the identical pipeline (matching -> threshold -> ledger) with a
rule-based extractor standing in for the model. No network call, no API key,
no variance between runs. If Gemini is rate-limited, the network drops, or a
key runs out mid-recording, this keeps the demo working.

It is not a full substitute for the model: it cannot read nuance the way an
LLM does, so its extraction is cruder (no contradiction detection, simpler
urgency reading). What it guarantees is that the safety-critical path —
threshold.py's rules — runs exactly the same way regardless of which
extractor produced the input, because that path never depended on the model
to begin with.
"""

from __future__ import annotations

import re

from . import pipeline, threshold, tools
from .schema import Category, ExtractedField, Extraction, IntakeKind

PLACES = (
    "kaloor", "fort cochin", "palarivattom", "aluva", "tripunithura",
    "perumbavoor", "angamaly", "kolenchery", "muvattupuzha", "thrissur",
)

ITEM_CATEGORY: tuple[tuple[str, Category, str], ...] = (
    ("wheelchair", Category.MOBILITY_AID, "wheelchair"),
    ("walker", Category.MOBILITY_AID, "walker"),
    ("oxygen concentrator", Category.RESPIRATORY, "oxygen concentrator"),
    ("oxygen", Category.RESPIRATORY, "oxygen concentrator"),
    ("nebulizer", Category.RESPIRATORY, "nebulizer"),
    ("hospital bed", Category.HOME_CARE, "hospital bed"),
    ("commode", Category.HOME_CARE, "commode chair"),
    ("air mattress", Category.HOME_CARE, "air mattress"),
    ("mattress", Category.HOME_CARE, "air mattress"),
    ("magnifier", Category.ASSISTIVE_TECH, "magnifier"),
    ("blood pressure", Category.DIGITAL_HEALTH, "blood pressure monitor"),
    ("bp monitor", Category.DIGITAL_HEALTH, "blood pressure monitor"),
    ("notebook", Category.SCHOOL_SUPPLIES, "notebook sets"),
    ("geometry box", Category.SCHOOL_SUPPLIES, "notebook sets"),
    ("rice", Category.FOOD, "rice"),
    ("bench", Category.FURNITURE, "friendship bench"),
)

SKILL_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("first aid", "first_aid"),
    ("nurse", "first_aid"),
    ("photography", "photography"),
    ("photo", "photography"),
    ("registration", "registration"),
    ("front desk", "registration"),
    ("translation", "translation"),
    ("interpreter", "translation"),
    ("drive", "driving"),
    ("driving", "driving"),
    ("logistics", "logistics"),
)

VOLUNTEER_SIGNAL = (
    "i can do", "i can help", "available to help", "put me on", "volunteer",
    "duty saturday", "duty on", "sign me up",
)
DONATION_SIGNAL = ("donat", "we'd like to give", "we have", "to give away")
OUT_OF_SCOPE_SIGNAL = ("tax receipt", "membership", "how do i join", "unsubscribe")

_QUANTITY_PATTERN = re.compile(r"\b(\d+)\b")


def _find_place(text: str) -> str:
    lowered = text.lower()
    for place in PLACES:
        if place in lowered:
            return place.title()
    return ""


def _find_item_category(text: str) -> tuple[Category | None, str]:
    lowered = text.lower()
    for keyword, category, item_label in ITEM_CATEGORY:
        if keyword in lowered:
            return category, item_label
    return None, ""


def _find_quantity(text: str) -> int:
    match = _QUANTITY_PATTERN.search(text)
    return int(match.group(1)) if match else 1


def deterministic_extract(message: str) -> Extraction:
    """Rule-based stand-in for extract.extract_with_rotation. Reuses
    threshold's own keyword scan for vulnerability, since that scan was
    always meant to run independently of the model."""
    category, item_label = _find_item_category(message)
    place = _find_place(message)
    quantity = _find_quantity(message)
    vulnerable_flags, evidence = threshold.detect_vulnerable_flags(message)

    urgency = "urgent" if "urgent" in message.lower() else "routine"

    extraction = Extraction(
        kind=IntakeKind.NEED,
        category=ExtractedField(
            value=category.value if category else None,
            confidence=0.85 if category else 0.0,
            source="message",
        ),
        item=ExtractedField(
            value=item_label or None,
            confidence=0.8 if item_label else 0.0,
        ),
        quantity=ExtractedField(value=quantity, confidence=0.9),
        location=ExtractedField(
            value=place or None, confidence=0.85 if place else 0.0
        ),
        urgency=ExtractedField(value=urgency, confidence=0.7),
        vulnerable_flags=vulnerable_flags,
        vulnerable_evidence=evidence,
        missing_fields=[] if category else ["category"],
        followup_questions=(
            [] if category else ["What item do you need?"]
        ),
    )
    return extraction


def _classify(message: str) -> str:
    lowered = message.lower()
    if any(signal in lowered for signal in OUT_OF_SCOPE_SIGNAL):
        return "out_of_scope"
    if any(signal in lowered for signal in DONATION_SIGNAL):
        return "donation"
    if any(signal in lowered for signal in VOLUNTEER_SIGNAL):
        return "volunteer"
    return "need"


def _find_shift_for_message(message: str):
    """Match a volunteer message to an open shift by required skill."""
    from . import store

    lowered = message.lower()
    matched_skill = None
    for keyword, skill in SKILL_KEYWORDS:
        if keyword in lowered:
            matched_skill = skill
            break

    shifts = store.load_shifts()
    if matched_skill:
        for shift in shifts:
            if shift.required_skill == matched_skill:
                return shift.id
    return shifts[0].id if shifts else None


def route_message(requester: str, message: str) -> dict:
    """The deterministic-mode counterpart to the agent's own routing: guess
    the kind from keywords, then call the same tool the agent would call."""
    kind = _classify(message)

    if kind == "donation":
        return tools.log_donation(requester, message)
    if kind == "volunteer":
        slot_id = _find_shift_for_message(message)
        if slot_id is None:
            return tools.escalate_to_human(
                "no_open_shift", "Volunteer signup with no matching open shift.",
                requester,
            )
        return tools.assign_shift(requester, slot_id, message)
    if kind == "out_of_scope":
        return tools.escalate_to_human(
            "out_of_scope",
            f"Not a need, donation, or volunteer signup: {message!r}",
            requester,
        )
    return tools.log_request(requester, message)


def process_need_deterministic(requester: str, message: str):
    return pipeline.process_need(requester, message, extractor=deterministic_extract)


def process_donation_deterministic(donor: str, message: str):
    return pipeline.process_donation(donor, message, extractor=deterministic_extract)
