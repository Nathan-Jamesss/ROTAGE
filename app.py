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
