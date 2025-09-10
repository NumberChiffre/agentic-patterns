import os
import json
import hashlib
import redis

client = None


def redis_cache_enabled() -> bool:
    return bool(os.getenv("REDIS_URL"))


def get_client():
    global client
    if client is None:
        url = os.getenv("REDIS_URL")
        if not url:
            raise RuntimeError("REDIS_URL not set")
        client = redis.from_url(url)
    return client


def make_preview_key(query: str, model: str, min_preview_tokens: int) -> str:
    h = hashlib.sha256((query or "").encode("utf-8")).hexdigest()
    return f"preview:{model}:{min_preview_tokens}:{h}"


def preview_cache_get(key: str) -> tuple[int, str] | None:
    if not redis_cache_enabled():
        return None
    c = get_client()
    val = c.get(key)
    if not val:
        return None
    try:
        data = json.loads(val)
        tokens = int(data.get("tokens", 0))
        text = str(data.get("text", ""))
        return tokens, text
    except Exception:
        return None


def preview_cache_set(key: str, tokens: int, text: str) -> None:
    if not redis_cache_enabled():
        return
    if tokens <= 0 or not text:
        return
    ttl = int(os.getenv("PREVIEW_CACHE_TTL", "600"))
    payload = json.dumps({"tokens": int(tokens), "text": text})
    c = get_client()
    c.setex(key, ttl, payload)


__all__ = [
    "redis_cache_enabled",
    "get_client",
    "make_preview_key",
    "preview_cache_get",
    "preview_cache_set",
]
