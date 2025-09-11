from __future__ import annotations

from typing import List
import os
import math
import random
import numpy as np
from openai import OpenAI


class LengthFeatures:
    def __init__(self, length_threshold: int = 2000) -> None:
        self.length_threshold = max(1, int(length_threshold))

    def compute(self, query: str) -> List[float]:
        length = len(query or "")
        word_count = len((query or "").split())
        return [
            1.0,
            min(1.0, length / float(self.length_threshold)),
            min(1.0, word_count / 100.0),
        ]


class EmbeddingFeatures:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        output_dim: int = 24,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.output_dim = max(8, int(output_dim))
        self.seed = int(seed)
        # Fixed random projection for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        # Assume source dim 1536; create projection matrix (output_dim x 1536)
        src_dim = 1536
        self.proj = np.random.normal(
            0, 1.0 / math.sqrt(src_dim), (self.output_dim, src_dim)
        )
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def compute(self, query: str) -> List[float]:
        text = (query or "").strip()
        if not text:
            return [0.0] * self.output_dim
        emb = self.client.embeddings.create(input=[text], model=self.model)
        vec = np.asarray(emb.data[0].embedding, dtype=np.float32)
        reduced = self.proj @ vec
        # z-score normalize reduced vector
        mu = float(np.mean(reduced))
        sigma = float(np.std(reduced) + 1e-6)
        reduced = (reduced - mu) / sigma
        return reduced.astype(np.float32).tolist()


def compute_intent_signals(query: str) -> List[float]:
    q = (query or "").lower()
    coding = (
        1.0
        if any(k in q for k in ["code", "python", "typescript", "bug", "compile"])
        else 0.0
    )
    reasoning = (
        1.0
        if any(k in q for k in ["prove", "explain", "reason", "why", "derive"])
        else 0.0
    )
    retrieval = (
        1.0
        if any(k in q for k in ["cite", "sources", "latest", "find", "search"])
        else 0.0
    )
    return [coding, reasoning, retrieval]


__all__ = ["LengthFeatures", "EmbeddingFeatures", "compute_intent_signals"]
