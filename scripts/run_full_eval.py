"""Stage 7 — multi-checkpoint full eval. Internal results only.

Runs the Stage 3.5 baseline_eval logic against N checkpoints sequentially
(concurrency=4 within each), writing one canonical results JSON.

Usage:
    uv run python scripts/run_full_eval.py \\
        --checkpoints "baseline,continual_phase1,continual_phase2,continual_phase3" \\
        --endpoints "http://ep1:8000/v1,http://ep2:8000/v1,http://ep3:8000/v1,http://ep4:8000/v1" \\
        --output results/full_eval.json

ZERO-FALLBACK: missing/mismatched checkpoints/endpoints → hard fail.
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


def aggregate_full_eval(
    per_checkpoint: dict[str, dict[str, Any]],
    ablation: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical Stage 7 results JSON structure (pure, testable)."""
    return {
        "checkpoints": list(per_checkpoint.keys()),
        "per_checkpoint": per_checkpoint,
        "ablations": ablation or {},
        "metadata": metadata,
    }


def write_full_eval(aggregated: dict[str, Any], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregated, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", required=True,
                   help="Comma-separated checkpoint names")
    p.add_argument("--endpoints", required=True,
                   help="Comma-separated inference endpoint URLs (parallel to --checkpoints)")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    checkpoints = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    if len(checkpoints) != len(endpoints):
        print(
            f"ERROR: --checkpoints (n={len(checkpoints)}) and --endpoints "
            f"(n={len(endpoints)}) must be parallel and non-empty",
            file=sys.stderr,
        )
        return 1

    per_checkpoint: dict[str, dict[str, Any]] = {}
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, endpoint in zip(checkpoints, endpoints):
        print(f"\n=== Eval checkpoint: {name} @ {endpoint} ===", flush=True)
        per_ckpt_temp = out_dir / f"_temp_{name}.json"
        result = await run_baseline(
            qwen_endpoint=endpoint,
            output_path=per_ckpt_temp,
            model=name,
            max_concurrency=args.max_concurrency,
            split=args.split,
        )
        per_checkpoint[name] = result

    test_split_n = next(iter(per_checkpoint.values()))["n_problems"]
    metadata = {
        "model": "Qwen3.5-7B",
        "test_split_n": test_split_n,
        "judge_model": "gemini-2.5-pro",
        "splits_seed": 42,
        "split": args.split,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    aggregated = aggregate_full_eval(per_checkpoint, ablation=None, metadata=metadata)
    write_full_eval(aggregated, args.output)
    print(f"\nWrote {args.output} with {len(checkpoints)} checkpoints.")
    return 0


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    sys.exit(main())
