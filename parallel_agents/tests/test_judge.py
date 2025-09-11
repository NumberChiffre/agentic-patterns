from __future__ import annotations


import pytest

from src.judging.judge import (
    _extract_json_object,
    judge_previews,
    compute_candidate_order,
)
from src.core.types import JudgeVerdict, JudgeScores


def test_extract_json_object_variants() -> None:
    assert _extract_json_object('{"a": 1}') == {"a": 1}
    assert _extract_json_object('noise {"a": 2} trailing') == {"a": 2}
    assert _extract_json_object("no json here") is None


@pytest.mark.asyncio
async def test_judge_previews_happy_path() -> None:
    verdict = await judge_previews(
        query="q",
        previews=["{}"],
        judge_model="gpt-FAKE",
        min_preview_tokens=5,
        num_candidates=1,
    )
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.winner_index == 0
    assert len(verdict.scores) == 1


def test_compute_candidate_order() -> None:
    v = JudgeVerdict(
        winner_index=1,
        scores=[
            JudgeScores(
                index=0, relevance=0.5, coverage=0.4, faithfulness=0.6, overall=0.5
            ),
            JudgeScores(
                index=1, relevance=0.9, coverage=0.8, faithfulness=0.9, overall=0.85
            ),
        ],
    )
    order = compute_candidate_order(v, total_candidates=2)
    assert order == [1, 0]
