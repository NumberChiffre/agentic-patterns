from __future__ import annotations

import pytest

from src.race.race import race_with_judge_and_stream
from src.core.types import Strategy


@pytest.mark.asyncio
async def test_race_with_judge_and_stream_baseline() -> None:
    idx, agent_name, debug = await race_with_judge_and_stream(
        query="what is test?",
        judge_model="gpt-FAKE",
        agent_models=["gpt-a", "gpt-b"],
        min_preview_tokens=5,
        strategy=Strategy.BASELINE,
    )
    assert 0 <= idx < 2
    assert isinstance(agent_name, str) and agent_name
    assert isinstance(debug, dict)
    assert "latencies_s" in debug
    assert "fallback_failed_indices" in debug


@pytest.mark.asyncio
async def test_race_with_judge_and_stream_bandit() -> None:
    idx, agent_name, debug = await race_with_judge_and_stream(
        query="what is test?",
        judge_model="gpt-FAKE",
        agent_models=["gpt-a", "gpt-b"],
        min_preview_tokens=5,
        strategy=Strategy.BANDIT,
        bandit_state_path=None,
    )
    assert 0 <= idx < 2
    assert isinstance(agent_name, str) and agent_name
    assert isinstance(debug, dict)
    assert "latencies_s" in debug
    assert "fallback_failed_indices" in debug
