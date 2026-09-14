"""CLI demo — processes the overnight inbox end to end and prints the
Daybreak queue. The fallback demo path if the Streamlit UI is unavailable
during recording; also useful for a quick sanity run after any change.

Usage:
    python main.py                     # live Gemini extraction, with key rotation
    python main.py --deterministic     # no LLM call, rule-based extraction
    python main.py --chat              # one message, streamed token-by-token,
                                        # for showing the agent is genuinely
                                        # live rather than scripted
"""

from __future__ import annotations

import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows terminals default to cp1252

from quartermaster import agent as agent_module
from quartermaster import ledger, llm, store, triage


def print_metrics_summary(agent) -> None:
    """Strands tracks real execution metrics on the agent itself — latency,
    token usage, per-tool call counts — cumulative across every message it's
    processed. Printing this is concrete evidence of genuine SDK usage rather
    than a claim: judges can see actual tool-call counts and token spend, not
    a description of them."""
    summary = agent.event_loop_metrics.get_summary()
    usage = summary.get("accumulated_usage", {})
    tool_usage = summary.get("tool_usage", {})

    print("=" * 60)
    print("SESSION METRICS (strands EventLoopMetrics)")
    print("-" * 60)
    print(f"  cycles: {summary.get('total_cycles', 0)}   "
          f"duration: {summary.get('total_duration', 0):.1f}s")
    print(f"  tokens: {usage.get('totalTokens', 0)} total "
          f"(in {usage.get('inputTokens', 0)} / out {usage.get('outputTokens', 0)})")
    if tool_usage:
        print("  tool calls:")
        for name, stats in tool_usage.items():
            exec_stats = stats.get("execution_stats", {})
            print(
                f"    {name}: {exec_stats.get('call_count', 0)} call(s), "
                f"{exec_stats.get('success_rate', 0) * 100:.0f}% success, "
                f"avg {exec_stats.get('average_time', 0):.2f}s"
            )


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

    print()
    print_metrics_summary(current_agent)


async def _stream_one_message(agent, prompt: str) -> None:
    """Stream a single response token-by-token via Strands' stream_async,
    for a close-up that shows the agent reasoning live rather than a
    scripted-looking instant reply. Tool invocations print inline when the
    stream surfaces them; everything else is the token stream itself."""
    async for event in agent.stream_async(prompt):
        if "data" in event:
            print(event["data"], end="", flush=True)
        elif "current_tool_use" in event:
            name = event["current_tool_use"].get("name")
            if name:
                print(f"\n  [tool: {name}]", end="", flush=True)
    print()


def _run_chat() -> None:
    """One message, streamed live. For the video: proof this is a real,
    responding agent, not a pre-recorded transcript."""
    if not llm.keys_configured():
        print("No GEMINI_API_KEY configured — --chat needs live extraction.")
        return

    current_agent = agent_module.build_agent(stream_externally=True)
    print("Quartermaster — live chat. One message, streamed as it's generated.")
    print("Type a message (or 'quit'):\n")

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in {"quit", "exit"}:
            break
        asyncio.run(_stream_one_message(current_agent, prompt))
        print()

    print_metrics_summary(current_agent)


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
    if "--chat" in sys.argv:
        store.reset()
        _run_chat()
        return

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
