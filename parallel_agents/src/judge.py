from __future__ import annotations

import json
import logging
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

import weave
from .factory import make_judge
from .instructions import JUDGE_INSTRUCTIONS_TEMPLATE
from .streaming import agentsdk_text_stream
from .types import JudgeVerdict

logger = logging.getLogger(__name__)


def _extract_json_object(text: str) -> dict[str, object] | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential_jitter(initial=0.2, max=2.5),
    stop=stop_after_attempt(3),
    reraise=True,
)
@weave.op
async def judge_previews(
    query: str,
    previews: list[str],
    judge_model: str,
    min_preview_tokens: int = 120,
    num_candidates: int | None = None,
) -> JudgeVerdict:
    logger.info("Starting judge evaluation of previews")
    for i, p in enumerate(previews):
        logger.debug(f"Preview {i}: {p[:min_preview_tokens]}...")
    judge_instructions = JUDGE_INSTRUCTIONS_TEMPLATE.format(
        num_candidates=num_candidates or len(previews)
    )
    judge = make_judge(judge_model, judge_instructions)
    payload: dict[str, object] = {
        "query": query,
        "candidates": [
            {"index": i, "preview_json": (p or "{}")} for i, p in enumerate(previews)
        ],
    }
    buf: list[str] = []
    async for chunk in agentsdk_text_stream(judge, json.dumps(payload)):
        buf.append(chunk)
    text = "".join(buf).strip()
    data = _extract_json_object(text)
    if not data:
        raise ValueError("Judge produced non-JSON output after retries")
    verdict = JudgeVerdict.model_validate(data)
    logger.info(f"Judge verdict: Winner is Candidate {verdict.winner_index}")
    for score in verdict.scores:
        logger.info(
            f"  Candidate {score.index} scores - Relevance: {score.relevance:.2f}, "
            f"Coverage: {score.coverage:.2f}, Faithfulness: {score.faithfulness:.2f}, "
            f"Overall: {score.overall:.2f}"
        )
    return verdict


def compute_candidate_order(verdict: JudgeVerdict, total_candidates: int) -> list[int]:
    index_to_overall: dict[int, float] = {
        s.index: float(s.overall) for s in verdict.scores
    }
    return sorted(
        range(total_candidates),
        key=lambda i: index_to_overall.get(i, 0.0),
        reverse=True,
    )
