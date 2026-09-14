"""CLI demo — processes the overnight inbox end to end and prints the
Daybreak queue. The fallback demo path if the Streamlit UI is unavailable
during recording; also useful for a quick sanity run after any change.

Usage:
    python main.py            # live Gemini extraction, with key rotation
    python main.py --deterministic   # no LLM call, rule-based extraction
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows terminals default to cp1252

from quartermaster import agent as agent_module
from quartermaster import ledger, llm, store, triage


def _run_live() -> None:
    """Drive the agent itself over the inbox, rotating keys on rate limits."""
    inbox = store.load_inbox()
    rotator = llm.rotator()
    current_agent = agent_module.build_agent()

    for entry in inbox:
        prompt = f"Message from {entry['requester']}: {entry['message']}"
        while True:
            try:
                current_agent(prompt)
                break
            except Exception as error:  # noqa: BLE001
                if llm.is_rate_limit(error) and rotator.rotate():
                    print(f"  (rate limited, switching to {rotator.label()})")
                    current_agent = agent_module.build_agent()
                    continue
                raise


def _run_deterministic() -> None:
    """Drive the pipeline directly with the rule-based extractor. No model
    call, no network — the demo's insurance policy."""
    from quartermaster import pipeline
    from quartermaster.deterministic import deterministic_extract, route_message

    pipeline.set_extractor(deterministic_extract)
    try:
        for entry in store.load_inbox():
            route_message(entry["requester"], entry["message"])
    finally:
        pipeline.reset_extractor()


def print_daybreak() -> None:
    decisions = store.load_decisions()
    summary = triage.summarize(decisions)

    print("=" * 60)
    print("QUARTERMASTER — Daybreak")
    print("HOPE Prime, Rotary District 3205")
    print("=" * 60)
    print(summary.headline())
    print()

    pending = triage.needs_human(decisions)
    if pending:
        print(f"NEEDS YOUR DECISION ({len(pending)})")
        print("-" * 60)
        for decision in pending:
            print(ledger.render_receipt(decision))
            print()
    else:
        print("Nothing needs you right now.")

    handled = triage.handled(decisions)
    if handled:
        print(f"HANDLED WHILE YOU SLEPT ({len(handled)})")
        print("-" * 60)
        for decision in handled:
            tag = decision.tier.value
            print(f"  [{tag}] {decision.requester}: {decision.raw_message[:60]}")


def main() -> None:
    deterministic = "--deterministic" in sys.argv
    store.reset()

    if deterministic:
        print("Running in deterministic mode — no model call, no network.\n")
        _run_deterministic()
    else:
        if not llm.keys_configured():
            print("No GEMINI_API_KEY configured. Falling back to deterministic mode.")
            print("Copy .env.example to .env and add a key to use live extraction.\n")
            _run_deterministic()
        else:
            _run_live()

    print()
    print_daybreak()


if __name__ == "__main__":
    main()
