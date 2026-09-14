"""Quartermaster — Daybreak.

The Streamlit UI. Daybreak, New Intake, and Pool call the pipeline directly
(tools.py) rather than routing through the conversational Agent: one model
call per message instead of two, which matters on a free-tier quota during a
live demo — this is where a coordinator would actually work day to day.

The Agent tab is different on purpose: it talks to the real Strands `Agent`
object (agent.py), unmodified, so a visitor can see it reason about a message
and decide which tool to call, rather than only ever seeing pre-decided
results.
"""

from __future__ import annotations

import os

import streamlit as st

# On Streamlit Community Cloud, keys are entered in the app's Secrets panel
# (TOML) and surface as st.secrets, not necessarily as real environment
# variables. Bridge them into os.environ before anything else loads, since
# llm.py reads keys via os.getenv() — this keeps local .env and cloud secrets
# working through the identical code path.
for _name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
    if _name in st.secrets and not os.getenv(_name):
        os.environ[_name] = st.secrets[_name]

from quartermaster import agent as agent_module
from quartermaster import deterministic, ledger, llm, pipeline, store, triage
from quartermaster.schema import HumanAction, Tier

st.set_page_config(page_title="Quartermaster", layout="wide", page_icon="📋")

TIER_STYLE = {
    Tier.SAFETY_HOLD: ("⛔", "#c0392b"),
    Tier.URGENT_INCOMPLETE: ("⚠", "#d68910"),
    Tier.CONTESTED: ("⚠", "#b9770e"),
    Tier.UNMATCHED: ("◻", "#7f8c8d"),
    Tier.AUTO_RESOLVED: ("✓", "#27ae60"),
    Tier.OUT_OF_SCOPE: ("·", "#95a5a6"),
}

CATEGORY_OPTIONS = [
    "mobility_aid", "respiratory", "home_care", "assistive_tech",
    "digital_health", "food", "school_supplies", "clothing", "transport",
    "hygiene", "books", "furniture", "other",
]


def _init_state() -> None:
    if "deterministic" not in st.session_state:
        # Default to deterministic regardless of whether a key is
        # configured. On a public deploy, Gemini's free-tier quota
        # (5 req/min) is shared across every visitor — a judge's first
        # click should never depend on whether someone else just used it up.
        # Live mode is one toggle away for anyone who wants to see it.
        st.session_state.deterministic = True
    if "seeded" not in st.session_state:
        store.reset()
        st.session_state.seeded = True
    _apply_mode()


def _apply_mode() -> None:
    if st.session_state.deterministic:
        pipeline.set_extractor(deterministic.deterministic_extract)
    else:
        pipeline.reset_extractor()


def _reset_demo() -> None:
    store.reset()
    st.session_state.pop("seeded", None)
    st.rerun()


def _run_overnight_inbox() -> None:
    for entry in store.load_inbox():
        deterministic.route_message(entry["requester"], entry["message"])
    st.rerun()


def sidebar() -> None:
    with st.sidebar:
        st.title("Quartermaster")
        st.caption("HOPE Prime · Rotary District 3205")
        st.caption(f"🤖 Strands Agent · {llm.MODEL_ID} · {len(agent_module.TOOLS)} tools")

        st.session_state.deterministic = st.toggle(
            "Deterministic mode",
            value=st.session_state.deterministic,
            help=(
                "No model call, no network — rule-based extraction. "
                "Gemini's free tier is 5 requests/minute, so this is the "
                "reliable choice for running the whole overnight inbox."
            ),
        )
        _apply_mode()

        if not llm.keys_configured() and not st.session_state.deterministic:
            st.warning("No GEMINI_API_KEY configured — falling back to deterministic.")
            st.session_state.deterministic = True
            _apply_mode()

        st.divider()

        if st.button("▶ Run overnight inbox", use_container_width=True):
            _run_overnight_inbox()

        if st.button("↺ Reset demo", use_container_width=True):
            _reset_demo()

        st.divider()
        decisions = store.load_decisions()
        summary = triage.summarize(decisions)
        st.metric("Arrived overnight", summary.total)
        st.metric("Handled", summary.handled)
        st.metric("Needs you", summary.needs_you)


def render_receipt_card(decision) -> None:
    icon, color = TIER_STYLE.get(decision.tier, ("·", "#888"))
    ts = decision.created_at.strftime("%H:%M") if decision.created_at else "--:--"

    st.markdown(
        f"<div style='border-left:4px solid {color};padding:0.5rem 1rem;"
        f"margin-bottom:0.5rem;background:rgba(127,127,127,0.06);'>"
        f"<b>{icon} TIER {decision.tier.value.split('_',1)[0]} · "
        f"{decision.tier.value.split('_',1)[1].replace('_',' ').upper()}</b>"
        f"&nbsp;&nbsp;<span style='opacity:0.7'>conf {decision.confidence:.2f} · {ts}</span>"
        f"<br>{decision.requester} — {decision.raw_message}"
        f"<br><i>{decision.escalation_reason or ''}</i>"
        f"</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 1, 1, 1, 3])
    for label, action, col in [
        ("Accept", HumanAction.ACCEPT, cols[0]),
        ("Edit", HumanAction.EDIT, cols[1]),
        ("Respond", HumanAction.RESPOND, cols[2]),
        ("Ignore", HumanAction.IGNORE, cols[3]),
    ]:
        if col.button(label, key=f"{decision.id}-{action.value}"):
            ledger.apply_human_action(decision.id, action)
            st.rerun()

    with st.expander("Show receipt"):
        st.code(ledger.render_receipt(decision), language=None)


def render_handled_row(decision) -> None:
    icon, _ = TIER_STYLE.get(decision.tier, ("·", "#888"))
    cols = st.columns([6, 1])
    cols[0].write(f"{icon} **{decision.requester}** — {decision.raw_message[:70]}")
    if decision.reversible and decision.outcome.value == "auto_resolved":
        if cols[1].button("Undo", key=f"undo-{decision.id}"):
            ledger.undo(decision.id)
            st.rerun()
    with st.expander("Receipt", expanded=False):
        st.code(ledger.render_receipt(decision), language=None)


def view_daybreak() -> None:
    decisions = store.load_decisions()
    summary = triage.summarize(decisions)

    st.markdown(f"### {summary.headline()}")
    st.divider()

    pending = triage.needs_human(decisions)
    if pending:
        st.subheader(f"Needs your decision ({len(pending)})")
        for decision in pending:
            render_receipt_card(decision)
    else:
        st.info("Nothing needs you right now.")

    handled = triage.handled(decisions)
    if handled:
        with st.expander(f"Handled while you slept ({len(handled)})", expanded=False):
            for decision in handled:
                render_handled_row(decision)


def view_intake() -> None:
    st.subheader("New message")
    st.caption(
        "Paste a message the way it would actually arrive, and watch "
        "Quartermaster decide."
    )

    kind = st.radio("This message is a", ["Need", "Donation", "Volunteer signup"], horizontal=True)
    requester = st.text_input("From", value="")
    message = st.text_area("Message", height=100)

    slot_id = None
    if kind == "Volunteer signup":
        shifts = store.load_shifts()
        slot_id = st.selectbox(
            "Which shift?",
            options=[s.id for s in shifts],
            format_func=lambda sid: next(
                f"{s.id} — {s.role} ({s.slot}, {len(s.assigned)}/{s.capacity})"
                for s in shifts if s.id == sid
            ),
        )

    if st.button("Process", type="primary"):
        if not requester or not message:
            st.error("Both fields are required.")
        else:
            with st.spinner("Extracting, matching, deciding..."):
                if kind == "Need":
                    decision = pipeline.process_need(requester, message)
                elif kind == "Donation":
                    decision = pipeline.process_donation(requester, message)
                else:
                    decision = pipeline.process_shift_signup(requester, message, slot_id)

            if decision.outcome.value == "auto_resolved":
                st.success(f"Auto-resolved — Tier {decision.tier.value}")
            else:
                st.warning(f"Escalated — {decision.escalation_reason}")
            st.code(ledger.render_receipt(decision), language=None)


LANDING_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Zilla+Slab:wght@400;600;700;900&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.qm-scope{ --paper:#EEE8D8; --paper-raised:#F6F1E4; --ink:#26221B; --ink-soft:#5B5442;
  --line:#C7B98F; --line-soft:#DCD3B4; --accent:#1F6F5C; --accent-strong:#154E41;
  --accent-soft:#DCE9E1; --brass:#8C6423; --safety:#A83A26; --safety-soft:#F2DCD4;
  max-width:800px; margin-inline:auto; display:block; }
@media (prefers-color-scheme: dark){
  .qm-scope{ --paper:#1A211E; --paper-raised:#212925; --ink:#E9E2CC; --ink-soft:#B4AB90;
    --line:#3C463F; --line-soft:#2C3530; --accent:#4CB79A; --accent-strong:#6FCFB3;
    --accent-soft:#20332C; --brass:#C79A54; --safety:#E2694F; --safety-soft:#3A2620; }
}

.qm-scope h1, .qm-scope h2, .qm-scope h3, .qm-scope h4{
  font-family:'Zilla Slab', Georgia, serif; color:var(--ink); font-weight:700; margin:0;
}
.qm-scope p, .qm-scope li{ color:var(--ink-soft); }
.qm-mono{ font-family:'IBM Plex Mono', ui-monospace, monospace; }

/* ---- Hero: name + tagline, nothing else ---- */
.qm-hero{ padding:64px 0 88px; text-align:left; }
.qm-hero .qm-badge{
  display:inline-flex; align-items:center; gap:8px;
  border:1.5px solid var(--accent); color:var(--accent-strong);
  padding:5px 13px; border-radius:3px; font-family:'IBM Plex Mono',monospace;
  font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase;
  transform:rotate(-1deg); margin-bottom:22px;
  animation:qmStamp 0.5s ease-out;
}
@keyframes qmStamp{ from{opacity:0; transform:rotate(-1deg) scale(1.25);} to{opacity:1; transform:rotate(-1deg) scale(1);} }
.qm-hero h1{ font-size:clamp(3rem,8vw,5.2rem); line-height:0.96; letter-spacing:-0.01em; }
.qm-hero .qm-tag{
  font-family:'Zilla Slab', serif; font-style:italic; color:var(--accent-strong);
  font-size:clamp(1.3rem,2.6vw,1.8rem); margin-top:14px;
}
.qm-scroll-hint{
  margin-top:38px; font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
  letter-spacing:0.1em; text-transform:uppercase; color:var(--ink-soft); opacity:0.6;
}

/* ---- Sections ---- */
.qm-section{ padding-block:40px; border-top:1px dashed var(--line); }
.qm-eyebrow{
  font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.14em;
  text-transform:uppercase; color:var(--accent); font-weight:600; margin-bottom:6px; display:block;
}

/* ---- Workflow ---- */
.qm-flow{display:flex; gap:0; align-items:stretch; margin:20px 0 14px; flex-wrap:wrap;}
.qm-step{
  flex:1 1 150px; border:1px solid var(--line); padding:16px 14px;
  text-align:center; background:var(--paper-raised);
}
.qm-step + .qm-step{ border-left:none; }
.qm-step .emoji{font-size:1.7rem; display:block; margin-bottom:6px;}
.qm-step .label{font-weight:700; font-size:0.92rem; color:var(--ink);}
.qm-step .desc{font-size:0.78rem; margin-top:4px;}
.qm-fork{display:flex; gap:10px; margin-top:12px;}
.qm-fork > div{flex:1; padding:14px; text-align:center; font-size:0.86rem; border-radius:2px;}
.qm-fork .ok{background:var(--accent-soft); border:1px solid var(--accent); color:var(--accent-strong);}
.qm-fork .stop{background:var(--safety-soft); border:1px solid var(--safety); color:var(--safety);}
@media (max-width: 700px){ .qm-flow{flex-direction:column;} .qm-step + .qm-step{border-left:1px solid var(--line); border-top:none;} }

/* ---- Cards ---- */
.qm-card{ border:1px solid var(--line); padding:16px 18px; margin-bottom:10px; background:var(--paper-raised); }
.qm-card b{display:block; margin-bottom:4px; color:var(--ink);}
.qm-next{ border:1px dashed var(--line); padding:16px 18px; font-size:0.92rem; }

/* ---- CTA gate ---- */
.qm-gate{ text-align:center; padding:48px 0 24px; }
.qm-gate p{ max-width:440px; margin:0 auto 18px; }
</style>
"""


def _landing_hero() -> None:
    st.markdown(LANDING_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="qm-scope">
          <div class="qm-hero">
            <span class="qm-badge">Rotary District 3205 · HOPE Prime</span>
            <h1>Quartermaster</h1>
            <div class="qm-tag">It knows what not to do.</div>
            <div class="qm-scroll-hint">↓ scroll for how it works</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _landing_body() -> None:
    # Each section is built as one complete, self-closed HTML string and
    # rendered in a single st.markdown() call. Streamlit wraps every call in
    # its own isolated container, so a <div> opened in one call and closed in
    # a later call never actually nests — the browser just auto-closes the
    # empty div at the end of that call's fragment. Splitting tags across
    # calls silently drops the styling on everything after the opening tag.

    mode = st.radio(
        "Explain it to me like I'm",
        ["🙂 Simple", "🛠 Technical"],
        horizontal=True,
        label_visibility="collapsed",
    )
    simple = mode.startswith("🙂")

    intro = (
        """
        Every day, people ask Rotary coordinators for help — a wheelchair,
        a hospital bed, someone to volunteer at an event. Right now, one
        person has to read every message and match it up by hand.
        <br><br>
        <b>Quartermaster does that automatically.</b> It reads each message,
        checks what's available, and connects the two — instantly, day or
        night. It only wakes up a real person when something needs a
        human judgment call: someone elderly or unwell, two people
        wanting the same thing, or something that just doesn't add up.
        """
        if simple
        else """
        Quartermaster is a <b>Strands agent</b> wired to <b>10 tools</b>. An LLM
        (Gemini) extracts structured, confidence-scored facts from each
        message. A separate, deterministic rules engine — no LLM in this
        step — decides whether to auto-resolve or escalate, and <i>why</i>.
        Every decision is written to an append-only ledger and can be undone.
        """
    )
    st.markdown(f'<div class="qm-scope qm-section">{intro}</div>', unsafe_allow_html=True)

    flow_steps = (
        """
        <div class="qm-flow">
          <div class="qm-step"><span class="emoji">📩</span><div class="label">Message comes in</div><div class="desc">"Need a wheelchair, Kaloor"</div></div>
          <div class="qm-step"><span class="emoji">🧠</span><div class="label">It reads it</div><div class="desc">Figures out what's needed</div></div>
          <div class="qm-step"><span class="emoji">🔍</span><div class="label">Checks what's available</div><div class="desc">Looks for a good match</div></div>
          <div class="qm-step"><span class="emoji">⚖️</span><div class="label">Decides</div><div class="desc">Safe to handle alone?</div></div>
        </div>
        <div class="qm-fork">
          <div class="ok"><b>✅ Yes</b><br>Matched automatically, no one has to lift a finger</div>
          <div class="stop"><b>🙋 No</b><br>Held for a coordinator, with the reason plainly stated</div>
        </div>
        """
        if simple
        else """
        <div class="qm-flow">
          <div class="qm-step"><span class="emoji">📩</span><div class="label">Message</div><div class="desc">Need, donation, shift signup</div></div>
          <div class="qm-step"><span class="emoji">🧬</span><div class="label">Extract</div><div class="desc">Gemini → confidence-scored fields</div></div>
          <div class="qm-step"><span class="emoji">📐</span><div class="label">Match</div><div class="desc">RapidFuzz, explained breakdown</div></div>
          <div class="qm-step"><span class="emoji">🧮</span><div class="label">Threshold</div><div class="desc">9 deterministic rules, T1–T9</div></div>
          <div class="qm-step"><span class="emoji">📒</span><div class="label">Ledger</div><div class="desc">Recorded, reversible, queued</div></div>
        </div>
        <div class="qm-fork">
          <div class="ok"><b>✅ AUTO_RESOLVED</b><br>Every rule passed</div>
          <div class="stop"><b>🙋 ESCALATED</b><br>One rule failed, reason attached</div>
        </div>
        """
    )
    st.markdown(
        f'<div class="qm-scope qm-section"><span class="qm-eyebrow">How it works</span>{flow_steps}</div>',
        unsafe_allow_html=True,
    )

    guarantee = (
        """
        <div class="qm-card">
        <b>It will never quietly hand out equipment to someone vulnerable.</b>
        If a message mentions someone elderly, unwell, a child, or an unsafe
        situation, Quartermaster doesn't just "try to be careful" — the
        option to act alone is taken off the table entirely before it can
        even consider it. A coordinator always makes that call.
        </div>
        """
        if simple
        else """
        <div class="qm-card">
        <b>Vulnerable cases are structurally, not behaviourally, blocked.</b>
        <code>filter_auto_matchable()</code> strips candidates from the
        list <i>before</i> the agent ever receives them — there is no
        actionable option for the model to reason past. Proven by
        <code>tests/test_threshold.py</code>: a dozen adversarial messages
        that must escalate, plus a control group of ordinary requests
        that must not be flagged.
        </div>
        """
    )
    st.markdown(
        f'<div class="qm-scope qm-section"><span class="qm-eyebrow">The guarantee</span>{guarantee}</div>',
        unsafe_allow_html=True,
    )

    grounding = """
        <div class="qm-card">
        Designed around <b>HOPE Prime</b>, Rotary District 3205's real
        2026–27 flagship project — the Palliative Equipment Library, which
        lends wheelchairs, walkers, hospital beds, oxygen concentrators, and
        nebulizers to families who need them temporarily. Seed data uses the
        district's real clubs and areas across the Cochin, Muvattupuzha, and
        Thrissur zones.
        </div>
        <div class="qm-next">
        <b>Next step:</b> piloting this with Rotary District 3205's leadership
        next term.
        </div>
        """
    st.markdown(
        f'<div class="qm-scope qm-section"><span class="qm-eyebrow">Built for a real program</span>{grounding}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="qm-scope qm-gate"><p>Try it on the seeded overnight inbox, '
        "or send it a message yourself.</p></div>",
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Enter the agent →", type="primary", use_container_width=True):
            st.session_state.entered = True
            st.rerun()

    st.markdown(
        '<div class="qm-scope" style="text-align:center; margin-top:18px;">'
        '<span class="qm-mono" style="font-size:0.8rem; opacity:0.6;">'
        "Agents for Humans Hackathon · Good Neighbor track · "
        '<a href="https://github.com/Nathan-Jamesss/ROTAGE" target="_blank">GitHub</a> · MIT'
        "</span></div>",
        unsafe_allow_html=True,
    )


def view_landing() -> None:
    _landing_hero()
    _landing_body()


def _send_to_agent_with_rotation(prompt: str) -> str:
    """Invoke the session's agent, rotating to the next Gemini key on a rate
    limit and rebuilding the agent before retrying — the same pattern
    main.py uses for the CLI. Without this, a single visitor tripping one
    key's 5-requests/minute limit would show an error to the next visitor
    too, even with nine other configured keys sitting unused."""
    rot = llm.rotator()
    while True:
        try:
            result = st.session_state.agent_instance(prompt)
            return str(result)
        except Exception as error:  # noqa: BLE001
            if llm.is_rate_limit(error) and rot.rotate():
                st.session_state.agent_instance = agent_module.build_agent()
                continue
            return f"(the agent hit an error: {error})"


def view_agent_console() -> None:
    st.subheader("🤖 Talk to the agent directly")
    st.caption(
        "This sends your message to the real Strands `Agent` object — the "
        "same one in `agent.py` — not a pre-wired form. It reads the "
        "message, decides what kind it is, and calls the matching tool "
        "itself. Daybreak and New Intake call the underlying pipeline "
        "directly to save quota; this tab is where you actually watch the "
        "agent think."
    )

    with st.expander("What the agent knows about itself"):
        st.write(f"**Model:** `{llm.MODEL_ID}` (Gemini, via Strands' native provider)")
        st.write(
            "**Tools:** "
            + ", ".join(f"`{t.tool_spec['name']}`" for t in agent_module.TOOLS)
        )
        st.write("**System prompt:**")
        st.code(agent_module.SYSTEM_PROMPT, language=None)

    if not llm.keys_configured():
        st.warning(
            "No GEMINI_API_KEY configured, so there's no live model to talk "
            "to. Daybreak and New Intake still work fully offline — this "
            "tab specifically needs a real Gemini call, since the agent's "
            "own decision about which tool to call always goes through the "
            "model, regardless of the deterministic toggle."
        )
        return

    st.caption(
        "⚠ Uses a live Gemini call per message (free tier: 5 requests/minute, "
        "shared across every visitor to this page). A couple of messages is plenty."
    )

    if "agent_instance" not in st.session_state:
        st.session_state.agent_instance = agent_module.build_agent()
        st.session_state.agent_transcript = []

    with st.form("agent_form", clear_on_submit=True):
        name = st.text_input("From")
        message = st.text_area("Message", height=80)
        sent = st.form_submit_button("Send to agent", type="primary")

    if sent:
        if not name or not message:
            st.error("Both fields are required.")
        else:
            with st.spinner("Agent is reading, deciding, and calling a tool..."):
                prompt = f"Message from {name}: {message}"
                reply = _send_to_agent_with_rotation(prompt)
            st.session_state.agent_transcript.append((name, message, reply))

    for who, said, reply in reversed(st.session_state.agent_transcript):
        st.markdown(f"**{who}:** {said}")
        st.markdown(f"**Quartermaster:** {reply}")
        st.divider()

    if st.session_state.agent_transcript:
        summary = st.session_state.agent_instance.event_loop_metrics.get_summary()
        usage = summary.get("accumulated_usage", {})
        tool_usage = summary.get("tool_usage", {})
        st.caption(
            f"This session so far: {summary.get('total_cycles', 0)} reasoning "
            f"cycles, {usage.get('totalTokens', 0)} tokens."
        )
        if tool_usage:
            st.write(
                {
                    name: stats.get("execution_stats", {}).get("call_count", 0)
                    for name, stats in tool_usage.items()
                }
            )

        decisions = store.load_decisions()
        if decisions:
            with st.expander("Receipt for the most recent decision"):
                st.code(ledger.render_receipt(decisions[-1]), language=None)


def view_pool() -> None:
    resources_tab, volunteers_tab, shifts_tab = st.tabs(
        ["Donations", "Volunteers", "Shifts"]
    )

    with resources_tab:
        resources = store.load_resources()
        st.dataframe(
            [
                {
                    "id": r.id, "item": r.item, "category": r.category.value,
                    "qty": r.quantity, "location": r.location, "donor": r.donor,
                    "available": r.available,
                }
                for r in resources
            ],
            use_container_width=True,
        )

    with volunteers_tab:
        volunteers = store.load_volunteers()
        st.dataframe(
            [
                {
                    "name": v.name, "skills": ", ".join(v.skills),
                    "location": v.location, "availability": ", ".join(v.availability),
                }
                for v in volunteers
            ],
            use_container_width=True,
        )

    with shifts_tab:
        shifts = store.load_shifts()
        st.dataframe(
            [
                {
                    "id": s.id, "role": s.role, "slot": s.slot,
                    "skill": s.required_skill, "assigned": f"{len(s.assigned)}/{s.capacity}",
                    "full": s.full,
                }
                for s in shifts
            ],
            use_container_width=True,
        )


def main() -> None:
    _init_state()

    if "entered" not in st.session_state:
        st.session_state.entered = False

    if not st.session_state.entered:
        # No sidebar, no tabs — just the landing page. A judge's first
        # impression is the name, the tagline, and a scroll, not a data table.
        view_landing()
        return

    sidebar()
    with st.sidebar:
        st.divider()
        if st.button("← Back to overview", use_container_width=True):
            st.session_state.entered = False
            st.rerun()

    tab_daybreak, tab_intake, tab_agent, tab_pool = st.tabs(
        ["📋 Daybreak", "✉ New Intake", "🤖 Agent", "📦 Pool"]
    )
    with tab_daybreak:
        view_daybreak()
    with tab_intake:
        view_intake()
    with tab_agent:
        view_agent_console()
    with tab_pool:
        view_pool()


if __name__ == "__main__":
    main()
