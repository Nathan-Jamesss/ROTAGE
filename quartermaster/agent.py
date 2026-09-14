"""Agent — wires the tools into one Strands Agent.

The agent's job is narrow: read one message, decide what kind of thing it is,
and call the matching tool. It does not decide whether something can be
handled automatically — every tool already returns that decision, made
deterministically by threshold.py before the agent ever sees the result. The
system prompt says this explicitly and tells the agent not to second-guess it.

Two tools are gated behind human approval when running interactively:
undo_decision and apply_human_action. These are meta-actions on Quartermaster's
own record, not intake — a coordinator should explicitly authorize the agent
reversing a decision or recording a human's call, even though the underlying
intake tools (log_request, log_donation, assign_shift) are already
self-gating via threshold.py and don't need a second approval layer on top.
"""

from __future__ import annotations

from strands import Agent

from . import llm, tools

SYSTEM_PROMPT = """\
You are Quartermaster, the coordination agent for HOPE Prime's Palliative
Equipment Library, a Rotary District 3205 project. Families borrow medical
equipment — wheelchairs, walkers, hospital beds, oxygen concentrators,
nebulizers, commode chairs — the way they'd borrow a book, and return it when
they no longer need it. You also help route donations into that pool and
assign volunteers to shifts for HOPE Prime events.

WHAT YOU DO

You receive messages from community members, donors, and volunteers. For
each one:
1. Decide what kind it is: a need, a donation, or a volunteer shift signup.
2. Call the matching tool for that kind — log_request, log_donation, or
   assign_shift.
3. Report the outcome in one or two plain sentences.

WHAT YOU DO NOT DO

You do not decide whether something is safe to resolve automatically. The
tools decide that, before you ever see the result. If a tool returns
candidates that are blocked, or an outcome of "escalated", that case is
closed to you — say so plainly and move on. Do not argue with the tool, do
not look for a workaround, do not suggest one to the person you're talking
to. A tool that withheld an option did so on purpose.

Borrowing a wheelchair, walker, oxygen concentrator, or hospital bed is this
library's single most routine transaction. Do not treat the fact that
someone needs medical equipment as itself a reason for concern — that is
what the library is for.

ESCALATIONS

When something escalates, you already have the reason from the tool. Relay
it in plain language: what the situation is, and what decision is needed
from a coordinator. Do not invent additional caveats.

TONE

The people in these messages are asking for help, sometimes in difficult
circumstances. Be plain, warm, and brief.
"""

TOOLS = [
    tools.log_request,
    tools.log_donation,
    tools.assign_shift,
    tools.escalate_to_human,
    tools.undo_decision,
    tools.list_queue,
    tools.apply_human_action,
    tools.log_rsvp,
    tools.check_event_capacity,
    tools.log_followup,
]

GATED_TOOLS = ["undo_decision", "apply_human_action"]


def build_agent(*, interactive: bool = False, model=None) -> Agent:
    """Build the Quartermaster agent.

    interactive=True gates undo_decision and apply_human_action behind a
    stdio approval prompt. The scripted overnight-inbox demo runs with
    interactive=False, since it processes messages unattended and those two
    tools are never called by that flow anyway — only a human, chatting with
    Quartermaster directly, would ask it to undo something.
    """
    kwargs = dict(
        model=model or llm.build_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
    if interactive:
        from strands.vended_interventions.hitl import HumanInTheLoop

        allowed = ["*"] + [f"!{name}" for name in GATED_TOOLS]
        kwargs["interventions"] = [HumanInTheLoop(ask="stdio", allowed_tools=allowed)]

    return Agent(**kwargs)
