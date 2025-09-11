from __future__ import annotations

from src.runtime.factory import make_candidate, make_full, make_judge
from src.core.types import JudgeVerdict


def test_make_candidate_uses_web_search_tool_by_default() -> None:
    agent = make_candidate("A", model="gpt-FAKE", instructions="do it")
    assert agent.name == "A"
    assert agent.model == "gpt-FAKE"
    assert agent.instructions == "do it"
    assert isinstance(agent.tools, list) and len(agent.tools) >= 1


def test_make_full_keeps_model_and_tools() -> None:
    base = make_candidate("A", model="gpt-FAKE", instructions="preview")
    full = make_full(base, instructions="full")
    assert base.model == full.model
    assert base.tools == full.tools
    assert full.name.endswith("(full)")


def test_make_judge_sets_output_type() -> None:
    judge = make_judge(model="gpt-FAKE", instructions="judge")
    assert judge.name == "Judge"
    assert judge.output_type is JudgeVerdict
