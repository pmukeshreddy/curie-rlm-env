"""Stage 7 — RLM scaffold on/off ablation.

Runs the final continual Phase 3 checkpoint TWO ways:
  Mode A — RLM ON  (sub_llm_max_turns=1, default RLM scaffold)
  Mode B — RLM OFF (sub_llm_max_turns=0, forces flat LM, no recursion)

Both modes use the same checkpoint. Difference is the sub-LM recursion budget.

Usage:
    uv run python scripts/run_rlm_ablation.py \\
        --checkpoint continual_phase3 \\
        --endpoint http://ep:8000/v1 \\
        --output results/rlm_ablation.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from curie_rlm_env.baseline_eval import TASK_IDS, run_baseline


def compute_delta(per_task_on: dict[str, Any], per_task_off: dict[str, Any]) -> dict[str, float]:
    """Per-task mean_reward delta (ON minus OFF). Tasks present in both maps."""
    delta: dict[str, float] = {}
    for task_id in TASK_IDS:
        on_mean = float(per_task_on.get(task_id, {}).get("mean_reward", 0.0))
        off_mean = float(per_task_off.get(task_id, {}).get("mean_reward", 0.0))
        delta[task_id] = round(on_mean - off_mean, 6)
    return delta


def aggregate_ablation(
    checkpoint: str,
    mode_on: dict[str, Any],
    mode_off: dict[str, Any],
) -> dict[str, Any]:
    """Build the ablation JSON structure (pure, testable)."""
    delta = compute_delta(mode_on["per_task"], mode_off["per_task"])
    return {
        "checkpoint": checkpoint,
        "mode_A_rlm_on": mode_on,
        "mode_B_rlm_off": mode_off,
        "delta_per_task": delta,
    }


def write_ablation(aggregated: dict[str, Any], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregated, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Checkpoint name (label only)")
    p.add_argument("--endpoint", required=True, help="Inference endpoint URL")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Mode A — RLM ON (sub_llm_max_turns=1) ===", flush=True)
    mode_on = await run_baseline(
        qwen_endpoint=args.endpoint,
        output_path=out_dir / "_temp_rlm_on.json",
        model=args.checkpoint,
        max_concurrency=args.max_concurrency,
        split=args.split,
        sub_llm_max_turns_override=1,
    )

    print(f"\n=== Mode B — RLM OFF (sub_llm_max_turns=0) ===", flush=True)
    mode_off = await run_baseline(
        qwen_endpoint=args.endpoint,
        output_path=out_dir / "_temp_rlm_off.json",
        model=args.checkpoint,
        max_concurrency=args.max_concurrency,
        split=args.split,
        sub_llm_max_turns_override=0,
    )

    aggregated = aggregate_ablation(args.checkpoint, mode_on, mode_off)
    write_ablation(aggregated, args.output)
    print(f"\nWrote {args.output}")
    print("\nDelta (ON - OFF) per task:")
    for task_id, d in aggregated["delta_per_task"].items():
        print(f"  {task_id:10s}  {d:+.4f}")
    return 0


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    sys.exit(main())
