from __future__ import annotations

from typing import Dict, List


class BaselineRouter:
    def __init__(self) -> None:
        pass

    def select(self, x: List[float], arms: List[str], k: int = 1) -> List[str]:
        k_eff = max(1, min(int(k), len(arms)))
        return list(arms)[:k_eff]

    def bulk_update(self, x: List[float], rewards: Dict[str, float]) -> None:
        # No-op
        return
