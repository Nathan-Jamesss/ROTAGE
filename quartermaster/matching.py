"""Matching — one engine, pointed in two directions.

A need searching the donation pool and a donation searching open needs are the
same operation. That is the observation the whole project rests on: what look
like five separate coordination problems are one problem with different entry
points.

Every score is decomposed, because a number a coordinator cannot interrogate is
a number they will not trust.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from .schema import Candidate, Category, Extraction, Request, Resource

W_CATEGORY = 50.0
W_ITEM = 30.0
W_QUANTITY = 10.0
W_LOCATION = 10.0

ADJACENT_CREDIT = 0.4

ADJACENT: set[frozenset[Category]] = {
    frozenset({Category.MOBILITY_AID, Category.HOME_CARE}),
    frozenset({Category.RESPIRATORY, Category.HOME_CARE}),
    frozenset({Category.ASSISTIVE_TECH, Category.DIGITAL_HEALTH}),
    frozenset({Category.SCHOOL_SUPPLIES, Category.BOOKS}),
    frozenset({Category.FOOD, Category.HYGIENE}),
    frozenset({Category.FURNITURE, Category.HOME_CARE}),
}


CATEGORY_FUZZY_FLOOR = 60.0


def _as_category(value) -> Category | None:
    """Parse a category value from the model. Exact spelling is asked for in
    the prompt, but models paraphrase ("mobility" for "mobility_aid"), so a
    fuzzy fallback stands behind the exact match rather than silently
    discarding the whole category signal — losing it costs half the score."""
    if isinstance(value, Category):
        return value
    if not value:
        return None

    normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return Category(normalized)
    except ValueError:
        pass

    best_category, best_score = None, 0.0
    for category in Category:
        score = fuzz.ratio(normalized, category.value)
        if score > best_score:
            best_category, best_score = category, score
    return best_category if best_score >= CATEGORY_FUZZY_FLOOR else None


def _category_score(wanted: Category | None, offered: Category | None) -> tuple[float, str]:
    if wanted is None or offered is None:
        return 0.0, "category unknown"
    if wanted is offered:
        return 1.0, f"exact ({wanted.value})"
    if frozenset({wanted, offered}) in ADJACENT:
        return ADJACENT_CREDIT, f"adjacent ({wanted.value} ~ {offered.value})"
    return 0.0, f"unrelated ({wanted.value} vs {offered.value})"


ITEM_NOISE_FLOOR = 50.0
SAME_CATEGORY_ITEM_CREDIT = 0.5


def _item_score(
    wanted: str, offered: str, same_category: bool
) -> tuple[float, str]:
    """Compare two item descriptions.

    partial_token_set_ratio is used because a short request ("blankets") has to
    find a longer inventory line ("12 woollen blankets, new"). Its floor for
    unrelated strings sits around 40, so anything below ITEM_NOISE_FLOOR is
    treated as no evidence at all rather than weak evidence.

    When the categories already match exactly, wording differences stop being
    informative: "school supplies" and "40 notebook sets" describe the same
    thing. So an exact category match sets a floor under the item score.
    """
    floor = SAME_CATEGORY_ITEM_CREDIT if same_category else 0.0

    if not wanted or not offered:
        return floor, "no item description, category carries the match" if floor else "no item description"

    raw = fuzz.partial_token_set_ratio(wanted.lower(), offered.lower())
    normalised = max(0.0, (raw - ITEM_NOISE_FLOOR) / (100.0 - ITEM_NOISE_FLOOR))

    if normalised >= floor:
        return normalised, f'"{wanted}" ~ "{offered}" ({raw:.0f}%)'
    return floor, f'wording differs, same category ({raw:.0f}% literal)'


def _quantity_score(needed, available: int) -> tuple[float, str]:
    try:
        needed_int = int(float(needed))
    except (TypeError, ValueError):
        return 0.5, "quantity unknown, partial credit"
    if needed_int <= 0:
        return 0.5, "quantity unclear"
    if available >= needed_int:
        return 1.0, f"{available} available >= {needed_int} needed"
    return available / needed_int, f"only {available} of {needed_int} needed"


def _location_score(wanted: str, offered: str) -> tuple[float, str]:
    if not wanted or not offered:
        return 0.3, "location unknown"
    a, b = wanted.strip().lower(), offered.strip().lower()
    if a == b:
        return 1.0, f"same area ({offered})"
    if fuzz.partial_ratio(a, b) > 80:
        return 0.6, f"nearby ({wanted} / {offered})"
    return 0.3, f"different area ({wanted} vs {offered})"


def score_pair(
    wanted_category: Category | None,
    wanted_item: str,
    wanted_quantity,
    wanted_location: str,
    resource: Resource,
) -> Candidate:
    """Score one need against one resource, keeping the full derivation."""
    cat, cat_why = _category_score(wanted_category, resource.category)
    item, item_why = _item_score(wanted_item, resource.item, same_category=cat == 1.0)
    qty, qty_why = _quantity_score(wanted_quantity, resource.quantity)
    loc, loc_why = _location_score(wanted_location, resource.location)

    breakdown = {
        "category_match": round(W_CATEGORY * cat, 1),
        "item_similarity": round(W_ITEM * item, 1),
        "qty_sufficiency": round(W_QUANTITY * qty, 1),
        "location_proximity": round(W_LOCATION * loc, 1),
    }

    return Candidate(
        resource_id=resource.id,
        label=f"{resource.item} — {resource.donor}",
        score=round(sum(breakdown.values()), 1),
        breakdown=breakdown,
        explanations=[
            f"category_match      {cat_why}",
            f"item_similarity     {item_why}",
            f"qty_sufficiency     {qty_why}",
            f"location_proximity  {loc_why}",
        ],
    )


def rank(extraction: Extraction, pool: list[Resource]) -> list[Candidate]:
    """A need looks through the donation pool."""
    wanted_category = _as_category(extraction.category.value)
    wanted_item = str(extraction.item.value or extraction.category.value or "")
    wanted_location = str(extraction.location.value or "")

    candidates = [
        score_pair(
            wanted_category,
            wanted_item,
            extraction.quantity.value,
            wanted_location,
            resource,
        )
        for resource in pool
        if resource.available
    ]
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def rank_reverse(extraction: Extraction, open_needs: list[Request]) -> list[Candidate]:
    """A donation looks through the open needs. Same engine, other direction."""
    offered_category = _as_category(extraction.category.value)
    offered_item = str(extraction.item.value or extraction.category.value or "")
    offered_location = str(extraction.location.value or "")

    try:
        offered_quantity = int(float(extraction.quantity.value))
    except (TypeError, ValueError):
        offered_quantity = 1

    candidates: list[Candidate] = []
    for need in open_needs:
        if need.status != "open":
            continue

        as_resource = Resource(
            id=need.id,
            donor=need.requester,
            category=_as_category(need.extraction.category.value) or Category.OTHER,
            item=str(need.extraction.item.value or ""),
            quantity=offered_quantity,
            location=str(need.extraction.location.value or ""),
        )
        candidate = score_pair(
            offered_category,
            offered_item,
            need.extraction.quantity.value,
            offered_location,
            as_resource,
        )
        candidate.label = f"{need.requester} — {need.extraction.item.value or 'request'}"
        candidates.append(candidate)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
