"""KeyRotator: cooldowns must expire, never accumulate into a permanent lock.

This is the mechanism a hosted deploy relies on to survive many visitors over
days without ever getting permanently stuck once every key has, at some
point, been rate-limited once.
"""

from __future__ import annotations

from quartermaster import llm


def _rotator(n: int) -> llm.KeyRotator:
    return llm.KeyRotator(keys=[f"key-{i}" for i in range(n)])


def test_rotate_moves_to_the_next_key() -> None:
    rot = _rotator(3)
    assert rot.current == "key-0"
    assert rot.rotate() is True
    assert rot.current == "key-1"


def test_rotate_wraps_around_once_the_first_cooldown_has_elapsed(monkeypatch) -> None:
    """Immediately re-rotating within the same cooldown window must NOT wrap
    back to a key still cooling down — that would defeat the whole point of
    tracking cooldowns instead of just cycling blindly."""
    now = [0.0]
    monkeypatch.setattr(llm.time, "monotonic", lambda: now[0])

    rot = _rotator(2)
    rot.rotate()
    assert rot.current == "key-1"

    now[0] += 70.0  # past key-0's cooldown
    assert rot.rotate() is True
    assert rot.current == "key-0"


def test_immediate_rewrap_fails_while_still_cooling_down() -> None:
    """The counterpart: with no time elapsed, wrapping to a key rotated away
    from moments ago must fail rather than pretend it's available again."""
    rot = _rotator(2)
    rot.rotate()  # key-0 now cooling down
    assert rot.rotate() is False  # key-1 cools down too; key-0 hasn't recovered yet


def test_all_keys_on_cooldown_reports_unavailable(monkeypatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(llm.time, "monotonic", lambda: now[0])

    rot = _rotator(2)
    rot.rotate()  # key-0 cools down, moves to key-1
    assert rot.rotate() is False  # key-1 cools down too, none left
    assert rot.available is False


def test_cooldown_expires_on_its_own(monkeypatch) -> None:
    """The actual bug this file exists to prevent: a rotator that never
    forgives a limit would stay permanently stuck once every key has been
    rate-limited at some point, even though each limit only lasts ~60s."""
    now = [1000.0]
    monkeypatch.setattr(llm.time, "monotonic", lambda: now[0])

    rot = _rotator(2)
    rot.rotate()  # key-0 on cooldown until 1000 + 65
    rot.rotate()  # key-1 on cooldown until 1000 + 65
    assert rot.available is False

    now[0] = 1000.0 + 66.0  # both cooldowns have now elapsed
    assert rot.available is True
    assert rot.rotate() is True  # can move again, not stuck forever


def test_retry_after_from_error_is_used_instead_of_default(monkeypatch) -> None:
    now = [0.0]
    monkeypatch.setattr(llm.time, "monotonic", lambda: now[0])

    rot = _rotator(2)
    rot.rotate(retry_after=5.0)
    assert rot.cooldown_until[0] == 5.0  # not the 65s default

    now[0] = 6.0
    assert rot.available is True


def test_parse_retry_after_reads_gemini_error_format() -> None:
    error = Exception(
        '{"error": {"message": "Please retry in 35.021852782s."}}'
    )
    assert llm._parse_retry_after(error) == 35.021852782


def test_parse_retry_after_returns_none_when_absent() -> None:
    assert llm._parse_retry_after(Exception("some other error")) is None


def test_single_key_rotate_always_fails() -> None:
    rot = _rotator(1)
    assert rot.rotate() is False


def test_ten_keys_supported() -> None:
    """The actual number the user needs for the hosted deploy."""
    rot = _rotator(10)
    seen = {rot.current}
    for _ in range(9):
        rot.rotate()
        seen.add(rot.current)
    assert len(seen) == 10


def test_label_reports_position() -> None:
    rot = _rotator(3)
    rot.rotate()
    assert rot.label() == "key 2 of 3"
