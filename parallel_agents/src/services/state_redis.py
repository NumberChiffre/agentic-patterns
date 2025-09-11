from __future__ import annotations

import json
import os
from typing import Any

from .cache_redis import redis_cache_enabled, get_client


DEFAULT_ROUTER_STATE_KEY = "router_state"


def router_state_enabled() -> bool:
    return redis_cache_enabled()


def _resolve_key(d: int) -> str:
    base = os.getenv("ROUTER_STATE_KEY", DEFAULT_ROUTER_STATE_KEY)
    return f"{base}:d{int(d)}"


def router_state_load(d: int) -> dict[str, Any] | None:
    if not router_state_enabled():
        return None
    c = get_client()
    key = _resolve_key(d)
    raw = c.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


def router_state_save(d: int, payload: dict[str, Any]) -> None:
    if not router_state_enabled():
        return
    c = get_client()
    key = _resolve_key(d)
    try:
        c.set(key, json.dumps(payload))
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "router_state_enabled",
    "router_state_load",
    "router_state_save",
]
