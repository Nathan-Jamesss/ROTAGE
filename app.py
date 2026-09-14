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


OVERVIEW_CSS = """
<style>
.qm-hero{padding:8px 0 4px;}
.qm-hero h1{font-size:2.6rem; margin-bottom:0; font-weight:800; letter-spacing:-0.01em;}
.qm-hero .sub{font-size:1.15rem; opacity:0.75; margin-top:4px;}
.qm-badge{
  display:inline-block; font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase;
  padding:4px 10px; border-radius:999px; border:1px solid rgba(127,127,127,0.35);
  opacity:0.75; margin-bottom:10px;
}
.qm-flow{display:flex; gap:8px; align-items:stretch; margin:22px 0 10px; flex-wrap:wrap;}
.qm-step{
  flex:1 1 140px; border:1px solid rgba(127,127,127,0.25); border-radius:10px;
  padding:16px 14px; text-align:center; background:rgba(127,127,127,0.04);
}
.qm-step .emoji{font-size:1.8rem; display:block; margin-bottom:6px;}
.qm-step .label{font-weight:700; font-size:0.95rem;}
.qm-step .desc{font-size:0.8rem; opacity:0.7; margin-top:4px;}
.qm-arrow{align-self:center; font-size:1.3rem; opacity:0.35; padding:0 2px;}
.qm-fork{display:flex; gap:10px; margin-top:10px;}
.qm-fork > div{flex:1; border-radius:10px; padding:14px; text-align:center; font-size:0.88rem;}
.qm-fork .ok{background:rgba(39,174,96,0.12); border:1px solid rgba(39,174,96,0.4);}
.qm-fork .stop{background:rgba(192,57,43,0.10); border:1px solid rgba(192,57,43,0.4);}
.qm-card{
  border:1px solid rgba(127,127,127,0.25); border-radius:10px; padding:16px 18px;
  margin-bottom:10px; background:rgba(127,127,127,0.03);
}
.qm-card b{display:block; margin-bottom:4px;}
.qm-next{
  border:1px dashed rgba(127,127,127,0.4); border-radius:10px; padding:16px 18px;
  font-size:0.92rem; opacity:0.85;
}
@media (max-width: 640px){ .qm-flow{flex-direction:column;} .qm-arrow{transform:rotate(90deg); align-self:center;} }
</style>
"""


def view_overview() -> None:
    st.markdown(OVERVIEW_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="qm-hero">
          <span class="qm-badge">Rotary District 3205 · HOPE Prime</span>
          <h1>Quartermaster</h1>
          <div class="sub">It knows what not to do.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Explain it to me like I'm",
        ["🙂 Simple", "🛠 Technical"],
        horizontal=True,
        label_visibility="collapsed",
    )
    simple = mode.startswith("🙂")

    if simple:
        st.markdown(
            """
            Every day, people ask Rotary coordinators for help — a wheelchair,
            a hospital bed, someone to volunteer at an event. Right now, one
            person has to read every message and match it up by hand.

            **Quartermaster does that automatically.** It reads each message,
            checks what's available, and connects the two — instantly, day or
            night. It only wakes up a real person when something needs a
            human judgment call: someone elderly or unwell, two people
            wanting the same thing, or something that just doesn't add up.
            """
        )
    else:
        st.markdown(
            """
            Quartermaster is a **Strands agent** wired to **10 tools**. An LLM
            (Gemini) extracts structured, confidence-scored facts from each
            message. A separate, deterministic rules engine — no LLM in this
            step — decides whether to auto-resolve or escalate, and *why*.
            Every decision is written to an append-only ledger and can be
            undone. See the **🤖 Agent** tab to talk to it directly, or the
            **📋 Daybreak** tab to see it working the overnight queue.
            """
        )

    st.markdown("#### How a message moves through it")
    if simple:
        st.markdown(
            """
            <div class="qm-flow">
              <div class="qm-step"><span class="emoji">📩</span><div class="label">Message comes in</div><div class="desc">"Need a wheelchair, Kaloor"</div></div>
              <div class="qm-arrow">→</div>
              <div class="qm-step"><span class="emoji">🧠</span><div class="label">It reads it</div><div class="desc">Figures out what's needed</div></div>
              <div class="qm-arrow">→</div>
              <div class="qm-step"><span class="emoji">🔍</span><div class="label">Checks what's available</div><div class="desc">Looks for a good match</div></div>
              <div class="qm-arrow">→</div>
              <div class="qm-step"><span class="emoji">⚖️</span><div class="label">Decides</div><div class="desc">Safe to handle alone?</div></div>
            </div>
            <div class="qm-fork">
              <div class="ok"><b>✅ Yes</b><br>Matched automatically, no one has to lift a finger</div>
              <div class="stop"><b>🙋 No</b><br>Held for a coordinator, with the reason plainly stated</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="qm-flow">
              <div class="qm-step"><span class="emoji">📩</span><div class="label">Message</div><div class="desc">Need, donation, or shift signup</div></div>
              <div class="qm-arrow">→</div>
              <div class="qm-step"><span class="emoji">🧬</span><div class="label">Extract</div><div class="desc">Gemini → confidence-scored fields</div></div>
              <div class="qm-arrow">→</div>
              <div class="qm-step"><span class="emoji">📐</span><div class="label">Match</div><div class="desc">RapidFuzz, explained breakdown</div></div>
              <div class="qm-arrow">→</div>
              <div class="qm-step"><span class="emoji">🧮</span><div class="label">Threshold</div><div class="desc">9 deterministic rules, T1–T9</div></div>
              <div class="qm-arrow">→</div>
              <div class="qm-step"><span class="emoji">📒</span><div class="label">Ledger</div><div class="desc">Recorded, reversible, queued</div></div>
            </div>
            <div class="qm-fork">
              <div class="ok"><b>✅ AUTO_RESOLVED</b><br>Every rule passed</div>
              <div class="stop"><b>🙋 ESCALATED</b><br>One rule failed, reason attached</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### The one guarantee that matters most")
    if simple:
        st.markdown(
            """
            <div class="qm-card">
            <b>It will never quietly hand out equipment to someone vulnerable.</b>
            If a message mentions someone elderly, unwell, a child, or an unsafe
            situation, Quartermaster doesn't just "try to be careful" — the
            option to act alone is taken off the table entirely before it can
            even consider it. A coordinator always makes that call.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="qm-card">
            <b>Vulnerable cases are structurally, not behaviourally, blocked.</b>
            <code>filter_auto_matchable()</code> strips candidates from the
            list <i>before</i> the agent ever receives them — there is no
            actionable option for the model to reason past. Proven by
            <code>tests/test_threshold.py</code>: a dozen adversarial messages
            that must escalate, plus a control group of ordinary requests
            that must not be flagged.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Built for a real program")
    st.markdown(
        """
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
        next term. If you're reviewing this as a coordinator or district
        officer and want to try it on real cases, the 🤖 Agent tab is the
        place to start.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Built for the Agents for Humans Hackathon — Good Neighbor Agents track · "
        "[GitHub](https://github.com/Nathan-Jamesss/ROTAGE) · MIT licensed"
    )


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
    sidebar()

    tab_overview, tab_daybreak, tab_intake, tab_agent, tab_pool = st.tabs(
        ["🏠 Overview", "📋 Daybreak", "✉ New Intake", "🤖 Agent", "📦 Pool"]
    )
    with tab_overview:
        view_overview()
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
