"""Model wiring, with key rotation.

Free-tier Gemini keys carry per-minute quotas. A single recording session can
trip one; a hosted app fielding visitors over days *will* trip several, and
each limit lifts on its own after about a minute. Keys are rotated on
rate-limit errors, and — critically for a long-lived server, as opposed to a
one-shot script — a rate-limited key is put on a cooldown timer rather than
retired forever. A rotator that never forgives a limit would eventually mark
every key exhausted and lock up permanently for the rest of the process's
life, which is exactly the "judges get stuck" failure this exists to prevent.

If every key is on cooldown, callers fall back to the deterministic pipeline
rather than failing: see quartermaster.deterministic.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "gemini-3.6-flash"

# Deterministic extraction. A demo that produces different output on each run
# is not a demo, it is a gamble.
PARAMS = {"temperature": 0.0, "max_output_tokens": 2048}

_RATE_LIMIT_MARKERS = (
    "rate limit", "quota", "resource_exhausted", "429",
    "too many requests", "exhausted",
)

# Gemini's free-tier window is 60s; pad slightly so a key isn't retried right
# at the edge of its own limit resetting.
DEFAULT_COOLDOWN_SECONDS = 65.0

_RETRY_AFTER_PATTERN = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def load_keys() -> list[str]:
    """Collect every configured key, in order, without duplicates. Supports
    GEMINI_API_KEY plus GEMINI_API_KEY_1 .. GEMINI_API_KEY_9 — up to 10 keys,
    enough for a judge to hammer the hosted app without hitting a dead end."""
    names = ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(1, 10)]
    keys: list[str] = []
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value and not value.startswith("your-key") and value not in keys:
            keys.append(value)
    return keys


def is_rate_limit(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _parse_retry_after(error: Exception) -> float | None:
    """Gemini's 429 body usually names its own retry delay, e.g. "Please
    retry in 35.02s". Use it when present; it's tighter than the flat
    default and gets a key back in rotation sooner."""
    match = _RETRY_AFTER_PATTERN.search(str(error))
    return float(match.group(1)) if match else None


@dataclass
class KeyRotator:
    """Hands out API keys, moving to the next one on a rate limit and
    putting the limited key on a cooldown timer rather than retiring it."""

    keys: list[str] = field(default_factory=load_keys)
    index: int = 0
    cooldown_until: dict[int, float] = field(default_factory=dict)

    def _on_cooldown(self, i: int, now: float) -> bool:
        return self.cooldown_until.get(i, 0.0) > now

    @property
    def available(self) -> bool:
        now = time.monotonic()
        return any(not self._on_cooldown(i, now) for i in range(len(self.keys)))

    @property
    def current(self) -> str:
        if not self.keys:
            raise RuntimeError(
                "No Gemini API key configured. Copy .env.example to .env and "
                "add GEMINI_API_KEY, or run Quartermaster in deterministic mode."
            )
        return self.keys[self.index]

    def rotate(self, retry_after: float | None = None) -> bool:
        """Put the current key on cooldown and move to the next key that
        isn't on one. Returns False when every key is currently cooling
        down — that's a "try again shortly" state, never a permanent one:
        cooldowns expire on their own, so a key rate-limited a minute ago is
        usable again without anyone restarting the process.
        """
        now = time.monotonic()
        self.cooldown_until[self.index] = now + (retry_after or DEFAULT_COOLDOWN_SECONDS)
        for offset in range(1, len(self.keys) + 1):
            nxt = (self.index + offset) % len(self.keys)
            if not self._on_cooldown(nxt, now):
                self.index = nxt
                return True
        return False

    def label(self) -> str:
        return f"key {self.index + 1} of {len(self.keys)}"


_rotator: KeyRotator | None = None


def rotator() -> KeyRotator:
    global _rotator
    if _rotator is None:
        _rotator = KeyRotator()
    return _rotator


def build_model(api_key: str | None = None):
    """Build a Gemini model bound to a specific key."""
    from strands.models.gemini import GeminiModel

    return GeminiModel(
        client_args={"api_key": api_key or rotator().current},
        model_id=MODEL_ID,
        params=PARAMS,
    )


def call_with_rotation(fn, *args, **kwargs):
    """Run fn, rotating keys on rate limits until one works or all are
    currently cooling down.

    fn receives the model as its first argument.
    """
    rot = rotator()
    last_error: Exception | None = None

    while rot.available:
        try:
            return fn(build_model(), *args, **kwargs)
        except Exception as error:  # noqa: BLE001 - provider errors vary
            last_error = error
            if not is_rate_limit(error) or not rot.rotate(_parse_retry_after(error)):
                raise

    raise RuntimeError(
        f"All {len(rot.keys)} Gemini keys are on cooldown right now. "
        f"They'll recover within a minute — switch on deterministic mode to "
        f"keep working meanwhile. Last error: {last_error}"
    )


def keys_configured() -> bool:
    return bool(load_keys())
