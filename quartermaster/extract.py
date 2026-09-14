"""Extraction — the model's entire job.

Quartermaster gives the model one narrow task: turn a messy inbound message
into structured facts, each with an honest confidence score. It is never asked
whether a request can be handled automatically. That question belongs to
threshold.py, which is deterministic and testable.

This split is deliberate. A model that both reads the message and decides its
fate has no auditable boundary; when it is wrong there is nothing to point at.
"""

from __future__ import annotations

from strands import Agent

from . import llm
from .schema import Extraction

EXTRACTION_PROMPT = """\
You read messages sent to the coordinator of HOPE Prime's Palliative
Equipment Library, a Rotary District 3205 project that lends medical
equipment — wheelchairs, walkers, hospital beds, oxygen concentrators,
nebulizers, commode chairs — to families who need it temporarily. You turn
messages into structured facts. You do not decide what happens next.

For each message, identify:
- kind: need (someone asking for help), donation (someone offering goods),
  volunteer (someone offering time), rsvp, or followup.
- category: exactly one of mobility_aid, respiratory, home_care,
  assistive_tech, digital_health, food, school_supplies, clothing, transport,
  hygiene, books, furniture, volunteer_time, other. Use this exact spelling,
  underscore included — not a paraphrase of it.
- item, quantity, unit, location, urgency, requester_name.
- vulnerable_flags: minor, elderly, medical, disability, safety, pregnancy.

CRITICAL — read this twice before setting any vulnerable_flags:

Every request in this system is inherently about medical equipment. Someone
asking for a wheelchair, a walker, an oxygen concentrator, or a hospital bed
is describing a completely routine transaction, not a vulnerability. Do NOT
set a flag just because:
- the item itself is medical equipment ("wheelchair" does not mean
  "disability", "oxygen concentrator" does not mean "medical" in the flagging
  sense — that's just what this library lends)
- the message mentions recovery, a procedure, or being unwell in a generic
  way ("recovering from surgery", "not feeling well")

DO set a flag only when the message states something beyond the routine
equipment need:
- minor: the person needing it is explicitly a child (not just "for my son"
  used as a routine stand-in — read for an actual child)
- elderly: an explicit age 65+, or words like grandmother/grandfather naming
  the person as elderly
- medical: a serious, specific condition — cancer, dialysis, bedridden,
  terminal, ICU, ventilator-dependent, hospice — not the mere fact of needing
  equipment or recovering from something ordinary
- disability: the message describes the person's condition (blind, deaf,
  cannot walk) as a fact about them, not the equipment they're requesting
- safety: an unsafe living situation, eviction, violence
- pregnancy: stated or clearly implied

When genuinely unsure, do not flag. A false negative here is recoverable —
a coordinator reviews escalations anyway when other rules catch something.
A false positive means a routine, working transaction can no longer help
someone, for no real reason.

Other rules you must follow:

1. Extract only what the message says. Never invent a quantity, a location, or
   a name that is not there. If it is absent, leave the value empty and list
   the field in missing_fields.

2. Give every field an honest confidence between 0 and 1. A low score is
   useful; a falsely high one is dangerous. Use the evidence field to quote the
   exact words that justified the value.

3. If two parts of the message disagree, record it in contradictions. Do not
   average them and do not silently choose one.

4. For each missing required field, write the specific question a coordinator
   should ask, in followup_questions. Never guess a default.

5. Quantities are plain integers. "a few" is not a quantity; leave it empty and
   ask.

Be literal. Someone is waiting on the other end of these messages.
"""


def extract(message: str, model=None) -> Extraction:
    """Pull structured facts out of one message using the model.

    Retries once on a transient malformed response (an occasional empty or
    invalid structured-output call) before giving up — this is a live model
    call in the critical path of every demo scenario, and a single retry is
    cheap insurance against a one-off hiccup.
    """
    agent = Agent(
        model=model or llm.build_model(),
        system_prompt=EXTRACTION_PROMPT,
    )
    try:
        return agent.structured_output(Extraction, message)
    except Exception as error:  # noqa: BLE001 - provider/validation errors vary
        if llm.is_rate_limit(error):
            raise
        return agent.structured_output(Extraction, message)


def extract_with_rotation(message: str) -> Extraction:
    """Extract, rotating API keys if one hits its rate limit."""
    return llm.call_with_rotation(lambda model: extract(message, model=model))
