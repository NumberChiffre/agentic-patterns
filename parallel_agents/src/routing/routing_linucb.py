from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ..services.state_redis import router_state_load, router_state_save

STATE_VERSION = 1


@dataclass
class _ArmState:
    # LinUCB with ridge regression and Sherman–Morrison updates
    A_inv: np.ndarray  # shape (d, d)
    b: np.ndarray  # shape (d,)


class LinUCBRouter:
    def __init__(
        self,
        d: int,
        alpha: float = 1.5,
        ridge_lambda: float = 1e-2,
        state_path: Path | None = None,
    ) -> None:
        if d <= 0:
            raise ValueError("d must be > 0")
        self.d = d
        self.alpha = float(alpha)
        self.ridge_lambda = float(ridge_lambda)
        self.state_path = state_path
        self._arms: dict[str, _ArmState] = {}
        # Try file state first when provided; otherwise, Redis-backed state if available
        if state_path and state_path.exists():
            self._load()
        else:
            data = router_state_load(self.d)
            if data:
                self._load_from_payload(data)

    def _ensure(self, arm: str) -> None:
        if arm in self._arms:
            return
        # Initialize A_inv = (lambda I)^-1 = (1/lambda) I, b = 0
        lam = max(1e-9, self.ridge_lambda)
        A_inv = (1.0 / lam) * np.eye(self.d, dtype=np.float64)
        b = np.zeros(self.d, dtype=np.float64)
        self._arms[arm] = _ArmState(A_inv=A_inv, b=b)

    def select(
        self,
        x: list[float],
        arms: list[str],
        k: int = 1,
        *,
        arm_bias: dict[str, float] | None = None,
    ) -> list[str]:
        if len(x) != self.d:
            raise ValueError("feature dimension mismatch")
        x_vec = np.asarray(x, dtype=np.float64)
        # Ensure all arms exist and collect states in a fixed order
        ordered_arms = list(arms)
        for arm in ordered_arms:
            self._ensure(arm)
        A_invs = np.stack(
            [self._arms[a].A_inv for a in ordered_arms], axis=0
        )  # (n, d, d)
        bs = np.stack([self._arms[a].b for a in ordered_arms], axis=0)  # (n, d)
        # theta_i = A_inv_i @ b_i  → (n, d)
        theta = (A_invs @ bs[..., None]).squeeze(-1)
        # mean_i = x^T theta_i → (n,)
        means = theta @ x_vec
        # Ax_i = A_inv_i @ x → (n, d); var_i = x^T Ax_i → (n,)
        Ax = A_invs @ x_vec
        vars_ = np.einsum("nd, d -> n", Ax, x_vec)
        vars_ = np.maximum(0.0, vars_)
        ucbs = self.alpha * np.sqrt(vars_)
        # Optional per-arm bias (e.g., latency penalty) added to scores
        if arm_bias:
            bias_vec = np.asarray([float(arm_bias.get(a, 0.0)) for a in ordered_arms])
        else:
            bias_vec = np.zeros_like(means)
        scores = means + ucbs + bias_vec
        # Top-k indices (descending)
        k_eff = max(1, min(k, scores.shape[0]))
        top_indices = np.argpartition(-scores, k_eff - 1)[:k_eff]
        # Order top-k by actual score descending
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [ordered_arms[i] for i in top_indices.tolist()]

    def update(self, x: list[float], arm: str, reward: float) -> None:
        if len(x) != self.d:
            raise ValueError("feature dimension mismatch")
        self._ensure(arm)
        st = self._arms[arm]
        # Sherman–Morrison: (A + x x^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        x_vec = np.asarray(x, dtype=np.float64)
        Ainv_x = st.A_inv @ x_vec
        denom = 1.0 + float(x_vec @ Ainv_x)
        if denom <= 1e-9:
            # Numerical safety: skip update if degenerate
            denom = 1e-9
        # A_inv = A_inv - (A_inv x)(A_inv x)^T / denom
        st.A_inv = st.A_inv - np.outer(Ainv_x, Ainv_x) / denom
        # b = b + reward * x
        st.b = st.b + (reward * x_vec)
        self._save()

    def bulk_update(self, x: list[float], rewards: dict[str, float]) -> None:
        if len(x) != self.d:
            raise ValueError("feature dimension mismatch")
        for arm, r in rewards.items():
            try:
                self.update(x, arm, float(r))
            except Exception:
                # Continue updating other arms even if one fails
                continue

    def decay(self, factor: float) -> None:
        """Apply exponential decay to accumulated evidence to adapt to drift.

        factor in (0,1]: lower means faster forgetting.
        """
        f = float(factor)
        if f <= 0:
            return
        for st in self._arms.values():
            st.A_inv = st.A_inv / f
            st.b = st.b * f
        self._save()

    def _save(self) -> None:
        payload = {
            "version": STATE_VERSION,
            "d": self.d,
            "arms": {
                arm: {
                    "A_inv": st.A_inv.tolist(),
                    "b": st.b.tolist(),
                }
                for arm, st in self._arms.items()
            },
        }
        # Save to file if configured
        if self.state_path:
            self.state_path.write_text(json.dumps(payload))
        # Also save to Redis if enabled
        router_state_save(self.d, payload)

    def _load(self) -> None:
        assert self.state_path is not None
        data = json.loads(self.state_path.read_text())
        self._load_from_payload(data)

    def _load_from_payload(self, data: dict) -> None:
        # Backward compatibility for old format
        if isinstance(data, dict) and "arms" not in data:
            for arm, st in data.items():
                A_inv = np.asarray(st.get("A_inv", []), dtype=np.float64)
                b = np.asarray(st.get("b", []), dtype=np.float64)
                if A_inv.size == 0 or b.size == 0:
                    continue
                self._arms[arm] = _ArmState(A_inv=A_inv, b=b)
            return
        # New format with version and dimension; reset on mismatch
        d_loaded = int(data.get("d", self.d))
        if d_loaded != self.d:
            self._arms = {}
            return
        arms = data.get("arms", {})
        for arm, st in arms.items():
            A_inv = np.asarray(st.get("A_inv", []), dtype=np.float64)
            b = np.asarray(st.get("b", []), dtype=np.float64)
            if A_inv.size == 0 or b.size == 0:
                continue
            self._arms[arm] = _ArmState(A_inv=A_inv, b=b)
