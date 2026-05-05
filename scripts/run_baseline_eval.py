"""CLI entrypoint for Stage 3.5 baseline eval.

Usage:
    uv run python scripts/run_baseline_eval.py \
        --endpoint $QWEN_ENDPOINT \
        --output results/baseline_qwen3_5_7b.json \
        --split test

Hard-fails at startup if QWEN_ENDPOINT is unreachable or GEMINI_API_KEY is unset.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

from curie_rlm_env.baseline_eval import TASK_IDS, run_baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3.5 baseline eval — Qwen3.5-7B + CurieRLMEnv."
    )
    p.add_argument(
        "--endpoint", required=True,
        help="Qwen vLLM OpenAI-compatible base URL (e.g. http://localhost:8000/v1)",
    )
    p.add_argument(
        "--output", required=True,
        help="Output JSON path (e.g. results/baseline_qwen3_5_7b.json)",
    )
    p.add_argument(
        "--model", default="Qwen/Qwen3-8B",
        help="Model name passed to the inference server.",
    )
    p.add_argument(
        "--max-concurrency", type=int, default=4,
        help="Max parallel rollouts (respect inference server limits).",
    )
    p.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Data split to evaluate (default: test). Requires data/curie/splits/{split}.jsonl.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    bars: dict[str, tqdm] = {}

    def progress(task_id: str, done: int, total: int) -> None:
        bar = bars.get(task_id)
        if bar is None:
            bar = tqdm(total=total, desc=task_id, position=TASK_IDS.index(task_id))
            bars[task_id] = bar
        bar.n = done
        bar.refresh()
        if done == total:
            bar.close()

    start = time.time()
    aggregated = asyncio.run(
        run_baseline(
            qwen_endpoint=args.endpoint,
            output_path=Path(args.output),
            model=args.model,
            max_concurrency=args.max_concurrency,
            progress=progress,
            split=args.split,
        )
    )
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s. Wrote {args.output}  (split={args.split})")
    print(f"n_problems: {aggregated['n_problems']}")
    print(f"overall mean reward: {aggregated['overall']['mean_reward']:.4f}")
    print("Per-task mean reward (n):")
    for task_id, stats in aggregated["per_task"].items():
        print(f"  {task_id:8s}  mean={stats['mean_reward']:.4f}  std={stats['std']:.4f}  n={stats['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
