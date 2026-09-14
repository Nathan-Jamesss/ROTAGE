"""Quartermaster — Daybreak.

The Streamlit UI. Calls the pipeline directly (tools.py) rather than routing
every action through the conversational Agent: one model call per message
instead of two, which matters on a free-tier quota during a live demo. The
Agent's own orchestration loop is what main.py demonstrates instead — this UI
is where a coordinator would actually work.
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

    tab_daybreak, tab_intake, tab_pool = st.tabs(["📋 Daybreak", "✉ New Intake", "📦 Pool"])
    with tab_daybreak:
        view_daybreak()
    with tab_intake:
        view_intake()
    with tab_pool:
        view_pool()


if __name__ == "__main__":
    main()
