from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dotenv import load_dotenv

from .race import race_with_judge_and_stream
from .types import Strategy

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run parallel race with judge and stream output")
    parser.add_argument("query", nargs="?", default=None, help="User query to research")
    parser.add_argument("--judge-model", dest="judge_model", default=None, help="Judge model id (env DEEP_RESEARCH_JUDGE_MODEL)")
    parser.add_argument("--agent-models", dest="agent_models", default=None, help="Comma-separated candidate model ids (env DEEP_RESEARCH_AGENT_MODELS)")
    parser.add_argument("--min-preview-tokens", dest="min_preview_tokens", type=int, default=None, help="Minimum preview tokens per candidate")
    parser.add_argument("--strategy", dest="strategy", choices=[Strategy.BASELINE, Strategy.BANDIT], default=Strategy.BASELINE, help="Routing strategy for candidate ordering")
    parser.add_argument("--bandit-alpha", dest="bandit_alpha", type=float, default=1.5, help="LinUCB alpha (exploration strength)")
    parser.add_argument("--bandit-ridge", dest="bandit_ridge", type=float, default=1e-2, help="LinUCB ridge regularization lambda")
    parser.add_argument("--bandit-state", dest="bandit_state", default=".router_state.json", help="Path to persist router state (JSON)")
    parser.add_argument("--bandit-length-threshold", dest="bandit_length_threshold", type=int, default=2000, help="Length threshold for feature normalization")
    parser.add_argument("--bandit-quality-weight", dest="bandit_quality_weight", type=float, default=0.8, help="Weight for judge quality in reward (0.0-1.0)")
    parser.add_argument("--bandit-speed-weight", dest="bandit_speed_weight", type=float, default=0.2, help="Weight for speed proxy in reward (0.0-1.0)")
    args = parser.parse_args(argv)

    def _parse_models(raw: str) -> list[str]:
        parts = [p.strip() for p in raw.split(",")]
        return [p for p in parts if p]
    
    agent_models = _parse_models(args.agent_models)
    winner_idx, winner_agent, debug = asyncio.run(
        race_with_judge_and_stream(
            query=args.query,
            judge_model=args.judge_model,
            agent_models=agent_models,
            min_preview_tokens=args.min_preview_tokens,
            tools=None,
            strategy=args.strategy,
            bandit_alpha=args.bandit_alpha,
            bandit_ridge_lambda=args.bandit_ridge,
            bandit_state_path=args.bandit_state,
            length_threshold=args.bandit_length_threshold,
            reward_weights=(args.bandit_quality_weight, args.bandit_speed_weight),
        )
    )

    result: dict[str, object] = {
        **debug,
        "winner_index": winner_idx,
        "winner_agent": winner_agent,
    }
    logger.info(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


