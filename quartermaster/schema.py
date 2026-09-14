"""Data models for Quartermaster.

Every fact the model pulls out of a message carries a confidence score and the
substring that justified it. Downstream code treats a low confidence the same
way it treats missing data.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class VulnerableFlag(str, Enum):
    MINOR = "minor"
    ELDERLY = "elderly"
    MEDICAL = "medical"
    DISABILITY = "disability"
    SAFETY = "safety"
    PREGNANCY = "pregnancy"


class Category(str, Enum):
    FOOD = "food"
    BEDDING = "bedding"
    SCHOOL_SUPPLIES = "school_supplies"
    CLOTHING = "clothing"
    MEDICAL = "medical"
    TRANSPORT = "transport"
    HYGIENE = "hygiene"
    BOOKS = "books"
    FURNITURE = "furniture"
    VOLUNTEER_TIME = "volunteer_time"
    OTHER = "other"


class IntakeKind(str, Enum):
    NEED = "need"
    DONATION = "donation"
    VOLUNTEER = "volunteer"
    RSVP = "rsvp"
    FOLLOWUP = "followup"


class Urgency(str, Enum):
    ROUTINE = "routine"
    SOON = "soon"
    URGENT = "urgent"


class Tier(str, Enum):
    SAFETY_HOLD = "1_safety_hold"
    URGENT_INCOMPLETE = "2_urgent_incomplete"
    CONTESTED = "3_contested"
    UNMATCHED = "4_unmatched"
    AUTO_RESOLVED = "5_auto_resolved"
    OUT_OF_SCOPE = "6_out_of_scope"


class Outcome(str, Enum):
    AUTO_RESOLVED = "auto_resolved"
    ESCALATED = "escalated"
    REVERSED = "reversed"


class HumanAction(str, Enum):
    ACCEPT = "accept"
    EDIT = "edit"
    RESPOND = "respond"
    IGNORE = "ignore"


class ExtractedField(BaseModel):
    """One fact pulled from free text, with the model's confidence in it."""

    value: str | int | float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Literal["message", "form", "both", "inferred"] = "message"
    evidence: str | None = None

    @property
    def present(self) -> bool:
        return self.value is not None and self.value != ""


class Extraction(BaseModel):
    """Everything the model is allowed to decide about one inbound message.

    Note what is absent: no verdict, no match, no judgement about whether this
    can be handled automatically. Those belong to threshold.py.
    """

    kind: IntakeKind = IntakeKind.NEED
    category: ExtractedField = Field(default_factory=ExtractedField)
    item: ExtractedField = Field(default_factory=ExtractedField)
    quantity: ExtractedField = Field(default_factory=ExtractedField)
    unit: ExtractedField = Field(default_factory=ExtractedField)
    location: ExtractedField = Field(default_factory=ExtractedField)
    urgency: ExtractedField = Field(default_factory=ExtractedField)
    requester_name: ExtractedField = Field(default_factory=ExtractedField)

    vulnerable_flags: list[VulnerableFlag] = Field(default_factory=list)
    vulnerable_evidence: str | None = None

    contradictions: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    followup_questions: list[str] = Field(default_factory=list)

    def field(self, name: str) -> ExtractedField:
        return getattr(self, name, ExtractedField())


class Resource(BaseModel):
    """Something available to give: a donation sitting in the pool."""

    id: str
    donor: str
    category: Category
    item: str
    quantity: int
    unit: str = "units"
    condition: str = "good"
    location: str = ""
    logged_at: datetime | None = None
    reserved_for: str | None = None

    @property
    def available(self) -> bool:
        return self.reserved_for is None


class Volunteer(BaseModel):
    id: str
    name: str
    skills: list[str] = Field(default_factory=list)
    availability: list[str] = Field(default_factory=list)
    phone: str = ""
    location: str = ""


class Shift(BaseModel):
    id: str
    project: str
    date: str
    slot: str
    role: str
    required_skill: str | None = None
    capacity: int = 1
    assigned: list[str] = Field(default_factory=list)

    @property
    def full(self) -> bool:
        return len(self.assigned) >= self.capacity


class Request(BaseModel):
    """A logged need, still open until something is matched to it."""

    id: str
    requester: str
    raw_message: str
    extraction: Extraction
    status: Literal["open", "matched", "escalated", "closed"] = "open"
    logged_at: datetime | None = None


class Candidate(BaseModel):
    """One possible match, with a fully explained score."""

    resource_id: str
    label: str = ""
    score: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)
    explanations: list[str] = Field(default_factory=list)
    blocked: bool = False
    blocked_reason: str | None = None


class RuleResult(BaseModel):
    """One rule that was evaluated, quoted literally. This is a receipt line."""

    rule: str
    passed: bool
    statement: str

    def render(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'}  {self.rule}: {self.statement}"


class Decision(BaseModel):
    """The complete, immutable record of one thing Quartermaster did."""

    id: str
    request_id: str
    created_at: datetime | None = None
    kind: IntakeKind = IntakeKind.NEED
    raw_message: str = ""
    requester: str = ""

    extraction: Extraction = Field(default_factory=Extraction)
    candidates: list[Candidate] = Field(default_factory=list)
    chosen: str | None = None

    rules: list[RuleResult] = Field(default_factory=list)
    outcome: Outcome = Outcome.ESCALATED
    tier: Tier = Tier.UNMATCHED
    confidence: float = 0.0
    escalation_reason: str | None = None

    reversible: bool = True
    reversed_at: datetime | None = None
    human_action: HumanAction | None = None
    human_note: str | None = None

    @property
    def needs_human(self) -> bool:
        return self.tier in {
            Tier.SAFETY_HOLD,
            Tier.URGENT_INCOMPLETE,
            Tier.CONTESTED,
            Tier.UNMATCHED,
        } and self.human_action is None
