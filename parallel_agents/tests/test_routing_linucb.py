from __future__ import annotations

from pathlib import Path


from src.routing_linucb import LinUCBRouter


def test_linucb_select_and_update(tmp_path: Path) -> None:
    state_file = tmp_path / "router_state.json"
    router = LinUCBRouter(d=2, alpha=1.0, ridge_lambda=1e-2, state_path=state_file)

    arms = ["a", "b", "c"]
    x = [1.0, 0.5]
    selected = router.select(x, arms, k=2)
    assert len(selected) == 2
    assert set(selected).issubset(set(arms))

    # Provide a positive reward for the first selected arm and negative for another
    router.update(x, selected[0], reward=1.0)
    router.update(x, selected[-1], reward=0.0)

    # After update, selection should still be valid and state persisted
    selected2 = router.select(x, arms, k=3)
    assert len(selected2) == 3
    assert state_file.exists()


def test_linucb_feature_dim_mismatch(tmp_path: Path) -> None:
    router = LinUCBRouter(d=3, state_path=tmp_path / "s.json")
    try:
        router.select([1.0, 2.0], ["x"], k=1)  # wrong dim
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_linucb_bulk_update(tmp_path: Path) -> None:
    state_file = tmp_path / "router_state.json"
    router = LinUCBRouter(d=2, alpha=1.0, ridge_lambda=1e-2, state_path=state_file)

    arms = ["a", "b", "c"]
    x = [0.2, 0.8]
    sel = router.select(x, arms, k=3)
    assert set(sel) == set(arms)

    rewards = {"a": 1.0, "b": 0.5, "c": 0.0}
    router.bulk_update(x, rewards)

    # After bulk update, state file should exist and selection is still valid
    sel2 = router.select(x, arms, k=2)
    assert len(sel2) == 2
    assert state_file.exists()
