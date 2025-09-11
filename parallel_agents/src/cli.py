from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dotenv import load_dotenv

import os
from datetime import datetime
import weave
import wandb
from agents import add_trace_processor
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor

from .race.race import race_with_judge_and_stream
from .core.types import Strategy
from .utils.citations import clean_citations_for_export


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run parallel race with judge and stream output"
    )
    parser.add_argument("query", nargs="?", default=None, help="User query to research")
    parser.add_argument(
        "--judge-model",
        dest="judge_model",
        default=None,
        help="Judge model id (env DEEP_RESEARCH_JUDGE_MODEL)",
    )
    parser.add_argument(
        "--agent-models",
        dest="agent_models",
        default=None,
        help="Comma-separated candidate model ids (env DEEP_RESEARCH_AGENT_MODELS)",
    )
    parser.add_argument(
        "--min-preview-tokens",
        dest="min_preview_tokens",
        type=int,
        default=None,
        help="Minimum preview tokens per candidate",
    )
    parser.add_argument(
        "--strategy",
        dest="strategy",
        choices=[Strategy.BASELINE, Strategy.BANDIT],
        default=Strategy.BASELINE,
        help="Routing strategy for candidate ordering",
    )
    parser.add_argument(
        "--bandit-alpha",
        dest="bandit_alpha",
        type=float,
        default=float(os.getenv("BANDIT_ALPHA", "1.5")),
        help="LinUCB alpha (exploration strength)",
    )
    parser.add_argument(
        "--bandit-ridge",
        dest="bandit_ridge",
        type=float,
        default=float(os.getenv("BANDIT_RIDGE", "1e-2")),
        help="LinUCB ridge regularization lambda",
    )
    parser.add_argument(
        "--bandit-state",
        dest="bandit_state",
        default=os.getenv("BANDIT_STATE", ".router_state.json"),
        help="Path to persist router state (JSON). If REDIS_URL is set, Redis is also used.",
    )
    parser.add_argument(
        "--bandit-length-threshold",
        dest="bandit_length_threshold",
        type=int,
        default=int(os.getenv("BANDIT_LENGTH_THRESHOLD", "2000")),
        help="Length threshold for feature normalization",
    )
    parser.add_argument(
        "--bandit-quality-weight",
        dest="bandit_quality_weight",
        type=float,
        default=float(os.getenv("BANDIT_QUALITY_WEIGHT", "0.8")),
        help="Weight for judge quality in reward (0.0-1.0)",
    )
    parser.add_argument(
        "--bandit-speed-weight",
        dest="bandit_speed_weight",
        type=float,
        default=float(os.getenv("BANDIT_SPEED_WEIGHT", "0.2")),
        help="Weight for speed proxy in reward (0.0-1.0)",
    )
    parser.add_argument(
        "--bandit-fallback-penalty",
        dest="bandit_fallback_penalty",
        type=float,
        default=float(os.getenv("BANDIT_FALLBACK_PENALTY", "0.05")),
        help="Penalty subtracted from reward when full run failed for a candidate (0.0-1.0)",
    )
    # Production tuning parameters
    parser.add_argument(
        "--adaptive-min-scale",
        dest="adaptive_min_scale",
        type=float,
        default=float(os.getenv("ADAPTIVE_MIN_SCALE", "0.75")),
        help="Lower multiplier bound for adaptive preview tokens",
    )
    parser.add_argument(
        "--adaptive-max-scale",
        dest="adaptive_max_scale",
        type=float,
        default=float(os.getenv("ADAPTIVE_MAX_SCALE", "1.50")),
        help="Upper multiplier bound for adaptive preview tokens",
    )
    parser.add_argument(
        "--latency-bias-scale",
        dest="latency_bias_scale",
        type=float,
        default=float(os.getenv("LATENCY_BIAS_SCALE", "0.05")),
        help="Magnitude of negative bias added to slower arms during selection",
    )
    parser.add_argument(
        "--speculative-min-query-length",
        dest="speculative_min_query_length",
        type=int,
        default=int(os.getenv("SPECULATIVE_MIN_QUERY_LENGTH", "2000")),
        help="Query length threshold to enable speculative top-2 full stage",
    )
    parser.add_argument(
        "--no-web-search",
        dest="no_web_search",
        action="store_true",
        help="Disable web search tool for candidates",
    )
    args = parser.parse_args(argv)

    def _parse_models(raw: str) -> list[str]:
        parts = [p.strip() for p in raw.split(",")]
        return [p for p in parts if p]

    project = os.getenv("WEAVE_PROJECT", "parallel-agents")
    weave.init(project)
    add_trace_processor(WeaveTracingProcessor())
    wandb.init(project=os.getenv("WANDB_PROJECT", project), reinit=True)

    agent_models = _parse_models(args.agent_models)
    race_kwargs = {
        "query": args.query,
        "judge_model": args.judge_model,
        "agent_models": agent_models,
        "tools": None,
        "enable_web_search": (not args.no_web_search),
        "strategy": args.strategy,
        "bandit_alpha": args.bandit_alpha,
        "bandit_ridge_lambda": args.bandit_ridge,
        "bandit_state_path": args.bandit_state,
        "length_threshold": args.bandit_length_threshold,
        "reward_weights": (args.bandit_quality_weight, args.bandit_speed_weight),
        "fallback_penalty": args.bandit_fallback_penalty,
        "adaptive_min_scale": args.adaptive_min_scale,
        "adaptive_max_scale": args.adaptive_max_scale,
        "latency_bias_scale": args.latency_bias_scale,
        "speculative_min_query_length": args.speculative_min_query_length,
    }
    if args.min_preview_tokens is not None:
        race_kwargs["min_preview_tokens"] = args.min_preview_tokens

    winner_idx, winner_agent, debug = asyncio.run(
        race_with_judge_and_stream(**race_kwargs)
    )

    full_response_text = str(debug.get("full_response", ""))
    logs_dir = os.path.abspath(os.path.join(os.getcwd(), "logs"))
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"response_{timestamp}.md"
    out_path = os.path.join(logs_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_response_text)

    export: dict[str, object] = {
        "winner_index": winner_idx,
        "winner_agent": winner_agent,
        "judge_scores": debug.get("judge_scores", []),
        "agent_models": debug.get("agent_models", []),
        "judge_model": debug.get("judge_model", ""),
        "strategy": str(debug.get("strategy", "")),
        "preview_tokens": debug.get("tokens", []),
        "full_response": full_response_text,
        "full_response_tokens": debug.get("full_response_tokens", 0),
        "citations": clean_citations_for_export(debug.get("citations", [])),
        "full_response_saved_to": out_path,
    }
    logger.info(json.dumps(export, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
