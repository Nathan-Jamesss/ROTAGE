# Quartermaster

**An autonomous coordinator for HOPE Prime's Palliative Equipment Library —
Rotary District 3205's project lending medical equipment to families who need
it temporarily, the way they'd borrow a book.**

Built for the [Agents for Humans Hackathon](https://agentsforhumans.devpost.com/)
(Good Neighbor Agents track), with the [Strands Agents SDK](https://strandsagents.com).

## What it does

Rotary coordinators field requests, donations, and volunteer signups all day
— by WhatsApp, by phone, by whoever's standing nearby. Every message gets
read, matched by hand against whatever's currently available, and followed
up on manually. It's slow, and the coordinator is the bottleneck for people
who are often asking for help at a difficult moment.

Quartermaster sits in that loop. It reads each message, tries to resolve it
on its own, and escalates to a human only when it genuinely can't — or
shouldn't — decide alone.

By morning, a coordinator opens **Daybreak** to something like:

```
14 arrived overnight    11 handled    3 need you
```

The three that need a human come with a full receipt: what was extracted,
which rule stopped it, and why.

## Who it's for

Rotary District 3205 coordinators running HOPE Prime — a real district
program, not a hypothetical — and the families, donors, and volunteers they
work with. The seed data uses the district's actual clubs (Rotary Club of
Kochi United, RC Cochin Knights, Angamaly Heritage, Perumbavoor Central...)
and real area names across the Cochin, Muvattupuzha, and Thrissur zones.

## The insight

Five things that look like separate problems — need-matching, donation
routing, volunteer shifts, event RSVPs, and follow-up — reduce to one
primitive: *something arrives, try to resolve it, escalate only when you
can't do it confidently.* Need-matching and donation-routing are literally
the same matching engine pointed in opposite directions against the same
pool. Volunteer shifts run the identical pattern against a different resource
type — time instead of goods. See [ARCHITECTURE.md](ARCHITECTURE.md) for the
full breakdown, including the two flows this build goes deep on versus the
two it deliberately stubs.

## The guarantee that matters most

**Quartermaster will not auto-assign resources in a case involving a
vulnerable person — and this is enforced structurally, not by asking the
model to be careful.**

A vulnerable case has its match candidates stripped out *before* the agent
ever sees them. There's nothing for the agent to act on, so no prompt and no
reasoning failure can produce an automatic assignment. This is proven, not
claimed:

```bash
pytest tests/test_threshold.py -v
```

runs a suite of adversarial messages (phrased to try to slip past a naive
filter) that must all escalate, alongside a control group of ordinary
equipment requests that must *not* — because in a medical equipment library,
over-flagging routine wheelchair requests would make the whole system
useless.

## How it works

```
message → extract structured facts (confidence-scored)
        → rank against the pool (RapidFuzz, explained score breakdown)
        → deterministic rules decide: auto-resolve or escalate
        → recorded in the ledger, reversible
```

The model's only job is extraction. Whether something can be auto-resolved
is decided by pure, unit-tested functions in `threshold.py` that never call
the model. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design,
including a mermaid diagram and every escalation rule.

## Setup

Requires Python 3.11+ and a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/Nathan-Jamesss/ROTAGE.git
cd ROTAGE
python -m venv .venv
.venv/Scripts/activate       # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
cp .env.example .env         # add your GEMINI_API_KEY
```

### Run the tests (no API key needed)

```bash
pytest -v
```

### Run the CLI demo

```bash
python main.py                  # live extraction via Gemini
python main.py --deterministic  # no model call, no network
```

### Run the UI

```bash
streamlit run app.py
```

Opens on the Daybreak queue. Use the sidebar to toggle deterministic mode,
run the seeded overnight inbox, or reset the demo to its starting state.

## Why Gemini, not Bedrock

Amazon Bedrock's model-access approval takes longer than this project's
timeline allowed for — a constraint at least two other teams in this
hackathon hit independently and solved the same way. Quartermaster uses
Strands' native Gemini provider instead, with automatic key rotation across
multiple API keys on rate-limit errors, since the free tier caps at 5
requests/minute.

## Designed to extend

`log_rsvp`, `check_event_capacity`, and `log_followup` exist as documented
tool signatures in `tools.py` but aren't built out for this submission —
believable RSVP and time-series follow-up data is hard to fabricate
honestly in the time available, and depth on the matching engine mattered
more than breadth across five shallow flows. The pattern for building them
out is identical to what's already here: a deterministic rule in
`threshold.py`, a tool wrapper, a place in the ledger.

Also not built: AgentCore deployment (optional per the hackathon rules;
strengthens the score but wasn't required), PII redaction on intake
messages, and full semantic (embedding-based) matching in place of the
current RapidFuzz scorer.

## Tech stack

Python, Strands Agents SDK, Gemini (native Strands provider), Pydantic v2,
RapidFuzz, Streamlit, pytest. JSON files for storage — inspectable,
diffable, and can't fail mid-demo the way a database connection can. Full
rationale for every choice, including what was deliberately *not* used, is
in [ARCHITECTURE.md](ARCHITECTURE.md).

## License

MIT. See [LICENSE](LICENSE).
