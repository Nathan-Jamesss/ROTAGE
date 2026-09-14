"""Store — reads seed data, holds runtime state, resets on demand.

data/ is the checked-in seed: donations, volunteers, shifts, the overnight
inbox. store/ is where a running session's state lives, seeded from data/ at
first use and rewritten as decisions happen. Resetting the demo means copying
data/ over store/ again; nothing in data/ is ever mutated.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Decision, Request, Resource, Shift, Volunteer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STORE_DIR = ROOT / "store"


def _read_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, rows: list[dict]) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)


def reset() -> None:
    """Re-seed all runtime state from data/. Wipes decisions and requests."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(STORE_DIR / "resources.json", _read_json(DATA_DIR / "resources.json"))
    _write_json(STORE_DIR / "volunteers.json", _read_json(DATA_DIR / "volunteers.json"))
    _write_json(STORE_DIR / "shifts.json", _read_json(DATA_DIR / "shifts.json"))
    _write_json(STORE_DIR / "requests.json", [])
    _write_json(STORE_DIR / "decisions.json", [])


def _ensure_seeded() -> None:
    if not (STORE_DIR / "resources.json").exists():
        reset()


def load_resources() -> list[Resource]:
    _ensure_seeded()
    return [Resource(**row) for row in _read_json(STORE_DIR / "resources.json")]


def load_volunteers() -> list[Volunteer]:
    _ensure_seeded()
    return [Volunteer(**row) for row in _read_json(DATA_DIR / "volunteers.json")]


def load_shifts() -> list[Shift]:
    _ensure_seeded()
    return [Shift(**row) for row in _read_json(STORE_DIR / "shifts.json")]


def load_requests() -> list[Request]:
    _ensure_seeded()
    return [Request(**row) for row in _read_json(STORE_DIR / "requests.json")]


def load_decisions() -> list[Decision]:
    _ensure_seeded()
    return [Decision(**row) for row in _read_json(STORE_DIR / "decisions.json")]


def load_inbox() -> list[dict]:
    """The overnight message queue. Read-only, never written back to."""
    return _read_json(DATA_DIR / "inbox.json")


def save_resources(resources: list[Resource]) -> None:
    _write_json(
        STORE_DIR / "resources.json",
        [r.model_dump(mode="json") for r in resources],
    )


def save_shifts(shifts: list[Shift]) -> None:
    _write_json(STORE_DIR / "shifts.json", [s.model_dump(mode="json") for s in shifts])


def save_requests(requests: list[Request]) -> None:
    _write_json(
        STORE_DIR / "requests.json", [r.model_dump(mode="json") for r in requests]
    )


def save_decisions(decisions: list[Decision]) -> None:
    _write_json(
        STORE_DIR / "decisions.json", [d.model_dump(mode="json") for d in decisions]
    )


def add_resource(resource: Resource) -> None:
    resources = load_resources()
    resources.append(resource)
    save_resources(resources)


def add_request(request: Request) -> None:
    requests = load_requests()
    requests.append(request)
    save_requests(requests)


def update_request(request: Request) -> None:
    requests = load_requests()
    for i, existing in enumerate(requests):
        if existing.id == request.id:
            requests[i] = request
            save_requests(requests)
            return
    raise KeyError(f"no request with id {request.id}")


def add_decision(decision: Decision) -> None:
    decisions = load_decisions()
    decisions.append(decision)
    save_decisions(decisions)


def update_decision(decision: Decision) -> None:
    decisions = load_decisions()
    for i, existing in enumerate(decisions):
        if existing.id == decision.id:
            decisions[i] = decision
            save_decisions(decisions)
            return
    raise KeyError(f"no decision with id {decision.id}")


def get_decision(decision_id: str) -> Decision | None:
    for decision in load_decisions():
        if decision.id == decision_id:
            return decision
    return None


def reserve_resource(resource_id: str, request_id: str) -> None:
    resources = load_resources()
    for resource in resources:
        if resource.id == resource_id:
            resource.reserved_for = request_id
            save_resources(resources)
            return
    raise KeyError(f"no resource with id {resource_id}")


def release_resource(resource_id: str) -> None:
    resources = load_resources()
    for resource in resources:
        if resource.id == resource_id:
            resource.reserved_for = None
            save_resources(resources)
            return
    raise KeyError(f"no resource with id {resource_id}")


def next_id(prefix: str, existing: list) -> str:
    """Sequential, readable ids: r-013, q-009, d-0007. Fine at demo scale."""
    numbers = []
    for item in existing:
        item_id = item.id if hasattr(item, "id") else item.get("id", "")
        digits = "".join(ch for ch in item_id.split("-")[-1] if ch.isdigit())
        if digits:
            numbers.append(int(digits))
    n = (max(numbers) + 1) if numbers else 1
    width = 4 if prefix == "d" else 3
    return f"{prefix}-{n:0{width}d}"
