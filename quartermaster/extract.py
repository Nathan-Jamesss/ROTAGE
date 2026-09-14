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
You read messages sent to a Rotary community coordinator and turn them into
structured facts. You do not decide what happens next.

For each message, identify:
- kind: need (someone asking for help), donation (someone offering goods),
  volunteer (someone offering time), rsvp, or followup.
- category, item, quantity, unit, location, urgency, requester_name.
- vulnerable_flags: minor, elderly, medical, disability, safety, pregnancy.
  Flag anything the message actually states or plainly implies. A mention of a
  child, an elderly relative, illness, disability, pregnancy, or an unsafe
  living situation all count.

Rules you must follow:

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
    """Pull structured facts out of one message using the model."""
    agent = Agent(
        model=model or llm.build_model(),
        system_prompt=EXTRACTION_PROMPT,
    )
    return agent.structured_output(Extraction, message)


def extract_with_rotation(message: str) -> Extraction:
    """Extract, rotating API keys if one hits its rate limit."""
    return llm.call_with_rotation(lambda model: extract(message, model=model))
