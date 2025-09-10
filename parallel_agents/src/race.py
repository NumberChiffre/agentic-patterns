from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from collections.abc import Callable

import weave
from .factory import make_candidate, make_full
from .instructions import FULL_RUN_INSTRUCTIONS_TEMPLATE, PREVIEW_INSTRUCTIONS_TEMPLATE
from .judge import compute_candidate_order, judge_previews
from .streaming import stream_response
from .routing_linucb import LinUCBRouter
from .types import PreviewOutcome, Strategy
from .citations import extract_citations_from_text, merge_and_dedupe, clean_citations_for_export

logger = logging.getLogger(__name__)


@weave.op
async def race_with_judge_and_stream(
    query: str,
    judge_model: str,
    agent_models: list[str],
    min_preview_tokens: int = 120,
    tools: list[object] | None = None,
    enable_web_search: bool = True,
    strategy: Strategy | str = Strategy.BASELINE,
    bandit_alpha: float = 1.5,
    bandit_ridge_lambda: float = 1e-2,
    bandit_state_path: str | None = ".router_state.json",
    feature_extractor: Callable[..., list[float]] | None = None,
    reward_weights: tuple[float, float] = (0.8, 0.2),
    length_threshold: int = 2000,
) -> tuple[int, str, dict[str, object]]:
    logger.info("=" * 80)
    logger.info("Starting parallel deep research run")
    logger.info(f"Query: {query}")
    logger.info(f"Min tokens per preview: {min_preview_tokens}")
    logger.info("=" * 80)
    if not agent_models:
        raise ValueError("agent_models must contain at least one model name")
    selected_models = list(agent_models)
    strategy_value = Strategy(str(strategy)) if not isinstance(strategy, Strategy) else strategy
    if strategy_value == Strategy.BANDIT:
        def _default_features(q: str, length_threshold: int = 2000) -> list[float]:
            length = len(q or "")
            # Add word count as a third feature to match existing state dimensions
            word_count = len((q or "").split())
            return [1.0, min(1.0, length / float(length_threshold)), min(1.0, word_count / 100.0)]
        router = LinUCBRouter(
            d=3,
            alpha=bandit_alpha,
            ridge_lambda=bandit_ridge_lambda,
            state_path=Path(bandit_state_path) if bandit_state_path else None,
        )
        feats = (feature_extractor or _default_features)(query, length_threshold=length_threshold)
        selected_models = router.select(feats, selected_models, k=len(selected_models))

    num_agents = len(selected_models)
    preview_instructions = [
        PREVIEW_INSTRUCTIONS_TEMPLATE.format(
            candidate_label=f"Candidate {chr(65 + i)}",
            num_candidates=num_agents,
            max_preview_tokens=min_preview_tokens,
            query=query,
        )
        for i in range(num_agents)
    ]
    agents = [
        make_candidate(
            name=f"Candidate {chr(65 + i)}[{selected_models[i]}]",
            model=selected_models[i],
            instructions=preview_instructions[i],
            tools=tools,
            enable_web_search=enable_web_search,
        )
        for i in range(num_agents)
    ]

    logger.info("Starting parallel preview generation")
    preview_tasks = [
        asyncio.create_task(stream_response(agent, query, stop_after_tokens=min_preview_tokens, capture_text=True, log_every_tokens=20))
        for agent in agents
    ]
    previews_results = await asyncio.gather(*preview_tasks)
    previews_outcomes = [
        PreviewOutcome(name=agents[i].name, text=(previews_results[i][1] or ""), tokens=previews_results[i][0])
        for i in range(len(agents))
    ]

    logger.info("All previews complete, sending to judge")
    previews_text = [po.text for po in previews_outcomes]
    verdict = await judge_previews(
        query,
        previews_text,
        judge_model,
        min_preview_tokens,
        num_candidates=num_agents,
    )
    ordered_indices = compute_candidate_order(verdict, len(agents))

    final_idx: int | None = None
    final_agent_name: str | None = None
    full_response_text: str = ""
    full_tokens: int = 0
    for idx in ordered_indices:
        selected_agent = agents[idx]
        selected_preview = previews_text[idx]
        logger.info(f"Selected candidate: {selected_agent.name}")
        full_instructions = FULL_RUN_INSTRUCTIONS_TEMPLATE.format(
            candidate_label=selected_agent.name,
            num_candidates=num_agents,
        )
        full_agent = make_full(selected_agent, full_instructions)
        seed = (
            f"User query:\n{query}\n\n"
            f"Your own winning preview JSON:\n{selected_preview}\n\n"
            f"Continue with the full answer now."
        )
        logger.info("Generating full answer")
        full_tokens, full_response_text = await stream_response(full_agent, seed, stop_after_tokens=None, capture_text=True, log_every_tokens=50)
        final_idx = idx
        final_agent_name = full_agent.name
        break

    if final_idx is None:
        raise RuntimeError("All candidates failed to stream full answer")

    # Bandit reward update (if enabled): blend judge overall with preview speed proxy
    if strategy_value == Strategy.BANDIT:
        wq, ws = reward_weights
        feats = (feature_extractor or _default_features)(query, length_threshold=length_threshold)
        for i, m in enumerate(selected_models):
            overall = 0.0
            for s in verdict.scores:
                if s.index == i:
                    overall = float(s.overall)
                    break
            speed_proxy = min(1.0, max(0.0, previews_outcomes[i].tokens / float(min_preview_tokens)))
            reward = max(0.0, min(1.0, wq * overall + ws * speed_proxy))
            router.update(feats, m, reward)

    debug: dict[str, object] = {
        "previews": previews_text,
        "tokens": [po.tokens for po in previews_outcomes],
        "judge_scores": [s.model_dump() for s in verdict.scores],
        "winner_index": final_idx,
        "winner_agent": final_agent_name,
        "full_response": full_response_text,
        "full_response_tokens": full_tokens,
        "agent_models": selected_models,
        "judge_model": judge_model,
        "strategy": strategy_value,
    }

    # Extract and dedupe citations from previews and full response text
    try:
        preview_citations = []
        for t in previews_text:
            preview_citations.extend(extract_citations_from_text(t or ""))
        full_citations = extract_citations_from_text(full_response_text or "")
        merged = merge_and_dedupe(preview_citations, full_citations)
        debug["citations"] = clean_citations_for_export(merged)
    except Exception:
        debug["citations"] = []

    logger.info("=" * 80)
    logger.info("Run complete!")
    logger.info("=" * 80)
    return final_idx, final_agent_name or "", debug


