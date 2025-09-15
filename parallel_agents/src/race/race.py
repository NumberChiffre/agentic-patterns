from __future__ import annotations

import asyncio
import json
import os
import logging
from pathlib import Path
from collections.abc import Callable
import time
import contextlib

import weave
import contextlib as _ctx

from ..runtime.factory import make_candidate, make_full
from ..core.instructions import (
    FULL_RUN_INSTRUCTIONS_TEMPLATE,
    PREVIEW_INSTRUCTIONS_TEMPLATE,
)
from ..judging.judge import compute_candidate_order, judge_previews
from ..runtime.streaming import stream_response
from ..runtime.metrics import (
    record_preview_latency,
    get_preview_latency_p95,
    compute_latency_norm,
)
from ..routing.routing_linucb import LinUCBRouter
from ..core.types import PreviewOutcome, Strategy
from ..core.types import RewardPolicy as RewardPolicyProtocol, FeatureExtractor as FeatureExtractorProtocol
from ..features import compute_context_features
from ..runtime.reward import QualityLatencyCostPolicy, QualityLatencyCostWeights
from ..utils.citations import (
    extract_citations_from_text,
    merge_and_dedupe,
    clean_citations_for_export,
)
from ..services.cache_redis import (
    make_preview_key,
    preview_cache_get,
    preview_cache_set,
)


class _NullTrace:
    def start(self, mark_as_current: bool | None = None) -> None:
        return None

    def finish(self, reset_current: bool | None = None) -> None:
        return None


def agents_trace(*_args, **_kwargs):  # type: ignore[override]
    return _NullTrace()


def agents_custom_span(_name: str):
    return _ctx.nullcontext()


logger = logging.getLogger(__name__)


def _resolve_strategy(value: Strategy | str) -> Strategy:
    return value if isinstance(value, Strategy) else Strategy(str(value))


def _default_features(q: str, length_threshold: int = 2000) -> list[float]:
    length = len(q or "")
    word_count = len((q or "").split())
    return [
        1.0,
        min(1.0, length / float(length_threshold)),
        min(1.0, word_count / 100.0),
    ]


def _adaptive_preview_tokens(
    query: str,
    base_min_preview_tokens: int,
    length_threshold: int,
    min_scale: float,
    max_scale: float,
) -> int:
    # Scale base_min_preview_tokens linearly by normalized query length in [min_scale, max_scale]
    q_len = len(query or "")
    norm = min(1.0, max(0.0, q_len / float(max(1, length_threshold))))
    scale = min_scale + (max_scale - min_scale) * norm
    scaled = int(round(base_min_preview_tokens * scale))
    return max(1, scaled)


def _select_models_for_strategy(
    query: str,
    candidate_models: list[str],
    strategy_value: Strategy,
    *,
    bandit_alpha: float,
    bandit_ridge_lambda: float,
    bandit_state_path: str | None,
    feature_extractor: Callable[..., list[float]],
    length_threshold: int,
) -> tuple[list[str], LinUCBRouter | None, list[float] | None]:
    selected_models = list(candidate_models)
    if strategy_value != Strategy.BANDIT:
        return selected_models, None, None

    feats = feature_extractor(query, length_threshold=length_threshold)
    router = LinUCBRouter(
        d=len(feats),
        alpha=bandit_alpha,
        ridge_lambda=bandit_ridge_lambda,
        state_path=Path(bandit_state_path) if bandit_state_path else None,
    )
    # Apply latency-aware bias if available (negative bias for slower arms)
    arm_bias: dict[str, float] = {}
    for m in selected_models:
        p95 = get_preview_latency_p95(m)
        if p95 and p95 > 0:
            lat_norm = compute_latency_norm(query, p95, length_threshold)
            arm_bias[m] = -_LATENCY_BIAS_SCALE * float(lat_norm)
    selected_models = router.select(
        feats, selected_models, k=len(selected_models), arm_bias=arm_bias or None
    )
    return selected_models, router, feats




def _collect_citations(previews_text: list[str], full_response_text: str) -> list[dict]:
    preview_citations = []
    for t in previews_text:
        preview_citations.extend(extract_citations_from_text(t or ""))
    full_citations = extract_citations_from_text(full_response_text or "")
    merged = merge_and_dedupe(preview_citations, full_citations)
    return clean_citations_for_export(merged)


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
    feature_extractor: Callable[..., list[float]] | FeatureExtractorProtocol | None = None,
    reward_weights: tuple[float, float] = (0.8, 0.2),
    bandit_cost_weight: float = 0.0,
    length_threshold: int = 2000,
    fallback_penalty: float = 0.05,
    adaptive_min_scale: float = 0.75,
    adaptive_max_scale: float = 1.50,
    latency_bias_scale: float = 0.05,
    speculative_min_query_length: int = 2000,
    preview_timeout_s: float | None = None,
    full_timeout_s: float | None = None,
    max_total_preview_tokens: int | None = None,
    max_total_full_tokens: int | None = None,
    max_total_cost_usd: float | None = None,
) -> tuple[int, str, dict[str, object]]:
    # Start a higher-level trace to group all nested agent runs under one workflow trace
    _trace = agents_trace("Parallel Deep Research Workflow")
    try:
        _trace.start(mark_as_current=True)
    except Exception:
        _trace = None
    logger.info("=" * 80)
    logger.info("Starting parallel deep research run")
    logger.info(f"Query: {query}")
    logger.info(f"Min tokens per preview: {min_preview_tokens}")
    logger.info("=" * 80)
    if not agent_models:
        raise ValueError("agent_models must contain at least one model name")
    strategy_value = _resolve_strategy(strategy)
    global _LATENCY_BIAS_SCALE
    _LATENCY_BIAS_SCALE = max(0.0, float(latency_bias_scale))
    # Support either callable or FeatureExtractor Protocol objects
    if feature_extractor and hasattr(feature_extractor, "compute"):
        feature_fn = lambda q, length_threshold: feature_extractor.compute(q)  # type: ignore[assignment]
    else:
        # Default to combined context features with optional embeddings controlled via env
        use_embedding = os.getenv("BANDIT_FEATURES", "length") == "embedding"
        emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        emb_dim = int(os.getenv("EMBEDDING_DIM", "24"))
        feature_fn = lambda q, length_threshold: compute_context_features(  # type: ignore[assignment]
            q,
            length_threshold=length_threshold,
            use_embedding=use_embedding,
            embedding_model=emb_model,
            embedding_output_dim=emb_dim,
        )
    selected_models, router, feats = _select_models_for_strategy(
        query,
        list(agent_models),
        strategy_value,
        bandit_alpha=bandit_alpha,
        bandit_ridge_lambda=bandit_ridge_lambda,
        bandit_state_path=bandit_state_path,
        feature_extractor=feature_fn,
        length_threshold=length_threshold,
    )

    num_agents = len(selected_models)
    adaptive_min_tokens = _adaptive_preview_tokens(
        query,
        min_preview_tokens,
        length_threshold,
        adaptive_min_scale,
        adaptive_max_scale,
    )
    preview_instructions = [
        PREVIEW_INSTRUCTIONS_TEMPLATE.format(
            candidate_label=f"Candidate {chr(65 + i)}",
            num_candidates=num_agents,
            max_preview_tokens=adaptive_min_tokens,
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
    # Attempt Redis preview cache; fall back to streaming per agent
    previews_results: list[tuple[int, str, float | None]] = []

    async def _timed_preview(agent_obj, q, stop_tokens, phase_name):
        t0 = time.perf_counter()
        coro = stream_response(
            agent_obj,
            q,
            stop_after_tokens=stop_tokens,
            capture_text=True,
            log_every_tokens=20,
            phase=phase_name,
        )
        if preview_timeout_s and preview_timeout_s > 0:
            tokens, text = await asyncio.wait_for(coro, timeout=preview_timeout_s)
        else:
            tokens, text = await coro
        elapsed = time.perf_counter() - t0
        return tokens, text, elapsed

    for i, agent in enumerate(agents):
        key = make_preview_key(query, selected_models[i], adaptive_min_tokens)
        cached = preview_cache_get(key)
        if cached is not None:
            previews_results.append((cached[0], cached[1], None))
        else:
            tokens, text, elapsed = await _timed_preview(
                agent,
                query,
                adaptive_min_tokens,
                f"preview:{agent.name}",
            )
            preview_cache_set(key, tokens, text)
            previews_results.append((tokens, text, elapsed))
    previews_outcomes = [
        PreviewOutcome(
            name=agents[i].name,
            text=(previews_results[i][1] or ""),
            tokens=previews_results[i][0],
            latency_s=previews_results[i][2],
        )
        for i in range(len(agents))
    ]
    # record latency metrics for models when available
    for i, po in enumerate(previews_outcomes):
        if po.latency_s and po.latency_s > 0:
            try:
                model_id = selected_models[i]
            except Exception:
                model_id = ""
            if model_id:
                record_preview_latency(model_id, float(po.latency_s))
    previews_text = [po.text for po in previews_outcomes]

    logger.info("All previews complete, sending to judge")
    with agents_custom_span("judge_previews"):
        verdict = await judge_previews(
            query,
            previews_text,
            judge_model,
            min_preview_tokens,
            num_candidates=num_agents,
        )
    ordered_indices = compute_candidate_order(verdict, num_agents)

    final_idx: int | None = None
    final_agent_name: str | None = None
    full_response_text: str = ""
    full_tokens: int = 0
    failed_full_indices: list[int] = []

    async def _run_full_for_index(i: int) -> tuple[int, int, str, str]:
        sel_agent = agents[i]
        sel_preview = previews_text[i]
        full_instr = FULL_RUN_INSTRUCTIONS_TEMPLATE.format(
            candidate_label=sel_agent.name,
            num_candidates=num_agents,
        )
        full_agent_local = make_full(sel_agent, full_instr)
        seed_local = (
            f"User query:\n{query}\n\n"
            f"Your own winning preview JSON:\n{sel_preview}\n\n"
            f"Continue with the full answer now."
        )
        coro = stream_response(
            full_agent_local,
            seed_local,
            stop_after_tokens=None,
            capture_text=True,
            log_every_tokens=50,
            phase="full_answer",
        )
        if full_timeout_s and full_timeout_s > 0:
            tokens_local, text_local = await asyncio.wait_for(coro, timeout=full_timeout_s)
        else:
            tokens_local, text_local = await coro
        return i, tokens_local, text_local, full_agent_local.name

    do_speculative = (len(query or "") >= speculative_min_query_length) and (
        len(ordered_indices) >= 2
    )
    if do_speculative:
        speculative_top_k = max(2, int(os.getenv("SPECULATIVE_TOP_K", "2")))
        speculative_top_k = min(speculative_top_k, len(ordered_indices))
        logger.info(f"Speculative top-{speculative_top_k} full stage enabled")
        topk = ordered_indices[:speculative_top_k]
        tasks = [asyncio.create_task(_run_full_for_index(i)) for i in topk]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        first = next(iter(done))
        try:
            idx_done, tok_done, txt_done, agent_name_done = await first
            final_idx = idx_done
            full_tokens = tok_done
            full_response_text = txt_done
            final_agent_name = agent_name_done
        except Exception as err:  # noqa: BLE001
            logger.info(f"Speculative winner failed: {err}")
        # cancel the other
        for p in pending:
            p.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await p
        if final_idx is None:
            # fallback to sequential over remaining
            for idx in ordered_indices:
                try:
                    idx2, tok2, txt2, agent2 = await _run_full_for_index(idx)
                    final_idx = idx2
                    full_tokens = tok2
                    full_response_text = txt2
                    final_agent_name = agent2
                    break
                except Exception:  # noqa: BLE001
                    failed_full_indices.append(idx)
                    continue
    else:
        for idx in ordered_indices:
            try:
                idx2, tok2, txt2, agent2 = await _run_full_for_index(idx)
                final_idx = idx2
                full_tokens = tok2
                full_response_text = txt2
                final_agent_name = agent2
                break
            except Exception:  # noqa: BLE001
                failed_full_indices.append(idx)
                continue

    if final_idx is None:
        raise RuntimeError("All candidates failed to stream full answer")

    if strategy_value == Strategy.BANDIT:
        # New reward policy with quality/latency/cost blend
        wq, ws = reward_weights
        weights = QualityLatencyCostWeights(
            quality_weight=float(wq),
            latency_weight=float(ws),
            cost_weight=float(bandit_cost_weight),
        )
        policy = QualityLatencyCostPolicy(
            weights=weights,
            fallback_penalty=fallback_penalty,
            length_threshold=length_threshold,
        )
        judge_overall = [float(s.overall) for s in verdict.scores]
        preview_tokens = [int(po.tokens) for po in previews_outcomes]
        rewards_by_model = policy.compute_rewards(
            query=query,
            models=selected_models,
            judge_overall=judge_overall,
            preview_tokens=preview_tokens,
            min_preview_tokens=adaptive_min_tokens,
            failed_full_indices=set(failed_full_indices or []),
        )
        if router is not None and feats is not None:
            router.bulk_update(feats, rewards_by_model)

    debug: dict[str, object] = {
        "previews": previews_text,
        "tokens": [po.tokens for po in previews_outcomes],
        "latencies_s": [po.latency_s for po in previews_outcomes],
        "judge_scores": [s.model_dump() for s in verdict.scores],
        "winner_index": final_idx,
        "winner_agent": final_agent_name,
        "full_response": full_response_text,
        "full_response_tokens": full_tokens,
        "agent_models": selected_models,
        "judge_model": judge_model,
        "strategy": strategy_value,
        "fallback_failed_indices": failed_full_indices,
    }

    debug["citations"] = _collect_citations(previews_text, full_response_text)
    logger.info("=" * 80)
    logger.info("Run complete!")
    logger.info("=" * 80)
    try:
        if _trace is not None:
            _trace.finish(reset_current=True)
    except Exception:
        pass
    return final_idx, final_agent_name or "", debug

