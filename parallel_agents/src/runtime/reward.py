from __future__ import annotations

import json
import os
from typing import Iterable

from pydantic import BaseModel, Field

from .metrics import get_preview_latency_p95, compute_latency_norm


class QualityLatencyCostWeights(BaseModel):
    quality_weight: float = Field(0.8, ge=0.0, le=1.0)
    latency_weight: float = Field(0.2, ge=0.0, le=1.0)
    cost_weight: float = Field(0.0, ge=0.0, le=1.0)

    def normalized(self) -> "QualityLatencyCostWeights":
        total = max(1e-9, float(self.quality_weight + self.latency_weight + self.cost_weight))
        return QualityLatencyCostWeights(
            quality_weight=self.quality_weight / total,
            latency_weight=self.latency_weight / total,
            cost_weight=self.cost_weight / total,
        )


def _load_price_table_from_env() -> dict[str, float]:
    raw = os.getenv("MODEL_PRICE_USD_PER_TOKEN_JSON", "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return {str(k): float(v) for k, v in data.items()}
    except Exception:
        return {}


_PRICE_TABLE_USD_PER_TOKEN: dict[str, float] = _load_price_table_from_env()


class QualityLatencyCostPolicy:
    def __init__(
        self,
        *,
        weights: QualityLatencyCostWeights | None = None,
        fallback_penalty: float = 0.05,
        length_threshold: int = 2000,
    ) -> None:
        self.weights = (weights or QualityLatencyCostWeights()).normalized()
        self.fallback_penalty = max(0.0, float(fallback_penalty))
        self.length_threshold = int(length_threshold)

    def _cost_norm(
        self,
        model: str,
        preview_tokens: int,
        min_preview_tokens: int,
    ) -> float:
        price = float(_PRICE_TABLE_USD_PER_TOKEN.get(model, 0.0))
        if price > 0.0:
            baseline = max(1e-9, price * float(min_preview_tokens))
            est_cost = price * float(max(0, preview_tokens))
            norm = min(1.0, max(0.0, est_cost / baseline))
            return 1.0 - norm
        return 1.0 - min(1.0, max(0.0, float(preview_tokens) / float(max(1, min_preview_tokens))))

    def _latency_term(self, query: str, model: str) -> float:
        p95 = get_preview_latency_p95(model)
        if p95 is None or p95 <= 0.0:
            return 0.5
        lat_norm = compute_latency_norm(query, p95, self.length_threshold)
        return 1.0 - lat_norm

    def compute_rewards(
        self,
        *,
        query: str,
        models: list[str],
        judge_overall: Iterable[float],
        preview_tokens: Iterable[int],
        min_preview_tokens: int,
        failed_full_indices: set[int] | None = None,
    ) -> dict[str, float]:
        jw = list(judge_overall)
        toks = list(preview_tokens)
        rewards: dict[str, float] = {}
        for i, model in enumerate(models):
            quality = float(jw[i] if i < len(jw) else 0.0)
            latency = float(self._latency_term(query, model))
            cost = float(self._cost_norm(model, int(toks[i] if i < len(toks) else 0), min_preview_tokens))
            r = (
                self.weights.quality_weight * quality
                + self.weights.latency_weight * latency
                + self.weights.cost_weight * cost
            )
            r = max(0.0, min(1.0, r))
            if failed_full_indices and i in failed_full_indices:
                r = max(0.0, r - self.fallback_penalty)
            rewards[model] = r
        return rewards


def estimate_token_cost_usd(model: str, tokens: int) -> float:
    try:
        price = float(_PRICE_TABLE_USD_PER_TOKEN.get(model, 0.0))
        return price * float(max(0, int(tokens)))
    except Exception:
        return 0.0



