"""Model wiring, with key rotation.

Free-tier Gemini keys carry per-minute quotas that a demo can exhaust in a
single pass over the overnight inbox. Several keys are loaded and rotated on
rate-limit errors so a recording session does not die halfway through.

If every key is exhausted, callers fall back to the deterministic pipeline
rather than failing: see quartermaster.deterministic.
"""

from __future__ import annotations

import os
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


def load_keys() -> list[str]:
    """Collect every configured key, in order, without duplicates."""
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


@dataclass
class KeyRotator:
    """Hands out API keys, moving on when one is rate-limited."""

    keys: list[str] = field(default_factory=load_keys)
    index: int = 0
    exhausted: set[int] = field(default_factory=set)

    @property
    def available(self) -> bool:
        return len(self.exhausted) < len(self.keys)

    @property
    def current(self) -> str:
        if not self.keys:
            raise RuntimeError(
                "No Gemini API key configured. Copy .env.example to .env and "
                "add GEMINI_API_KEY, or run Quartermaster in deterministic mode."
            )
        return self.keys[self.index]

    def rotate(self) -> bool:
        """Mark the current key exhausted and move to the next usable one.

        Returns False when every key is spent.
        """
        self.exhausted.add(self.index)
        for offset in range(1, len(self.keys) + 1):
            nxt = (self.index + offset) % len(self.keys)
            if nxt not in self.exhausted:
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
    """Run fn, rotating keys on rate limits until one works or all are spent.

    fn receives the model as its first argument.
    """
    rot = rotator()
    last_error: Exception | None = None

    while rot.available:
        try:
            return fn(build_model(), *args, **kwargs)
        except Exception as error:  # noqa: BLE001 - provider errors vary
            last_error = error
            if not is_rate_limit(error) or not rot.rotate():
                raise

    raise RuntimeError(
        f"All {len(rot.keys)} Gemini keys are rate-limited. "
        f"Switch on deterministic mode to keep working. Last error: {last_error}"
    )


def keys_configured() -> bool:
    return bool(load_keys())
