from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field


DEFAULT_METRICS_PATH = ".router_metrics.json"
MAX_LAT_SAMPLES = 100


class PreviewLatencyMetrics(BaseModel):
    preview_latency_s: dict[str, list[float]] = Field(default_factory=dict)


def _resolve_path(path: str | None) -> Path:
    p = Path(path or os.getenv("ROUTER_METRICS_PATH", DEFAULT_METRICS_PATH))
    return p


def _load(path: str | None = None) -> PreviewLatencyMetrics:
    p = _resolve_path(path)
    if not p.exists():
        return PreviewLatencyMetrics()
    try:
        data = json.loads(p.read_text())
        return PreviewLatencyMetrics(**data)
    except Exception:
        return PreviewLatencyMetrics()


def _save(store: PreviewLatencyMetrics, path: str | None = None) -> None:
    p = _resolve_path(path)
    try:
        p.write_text(store.model_dump_json())
    except Exception:
        pass


def record_preview_latency(
    model: str, latency_s: float, path: str | None = None
) -> None:
    if not model or not isinstance(latency_s, (int, float)):
        return
    if latency_s <= 0:
        return
    store = _load(path)
    arr = list(store.preview_latency_s.get(model, []))
    arr.append(float(latency_s))
    if len(arr) > MAX_LAT_SAMPLES:
        arr = arr[-MAX_LAT_SAMPLES:]
    store.preview_latency_s[model] = arr
    _save(store, path)


def get_preview_latency_p95(model: str, path: str | None = None) -> float | None:
    store = _load(path)
    arr = store.preview_latency_s.get(model)
    if not arr:
        return None
    try:
        q = float(np.percentile(np.asarray(arr, dtype=float), 95))
        return q
    except Exception:
        return None


def compute_latency_norm(
    query: str, p95_s: float | None, length_threshold: int
) -> float:
    if p95_s is None or p95_s <= 0:
        return 0.0
    try:
        norm_len = min(
            1.0, max(0.0, len(query or "") / float(max(1, length_threshold)))
        )
    except Exception:
        norm_len = 0.0
    # Increase base allowance for longer queries; base window 3s..6s
    base = 3.0 + 3.0 * norm_len
    return max(0.0, min(1.0, p95_s / base))
