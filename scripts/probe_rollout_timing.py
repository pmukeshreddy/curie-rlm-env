"""Stage 5b empirical pre-flight probe.

Measures wall-clock per rollout on 5 rollouts each for HFE (free-form),
DFT-S (retrieval w/ LLMSim), and BIOGR (geometric). Numbers feed into
the final Stage 5b TOMLs (max_steps, rollouts_per_example).

Usage:
    uv run python scripts/probe_rollout_timing.py --endpoint $QWEN_ENDPOINT

Hard-fails (ZERO-FALLBACK) if QWEN endpoint or GEMINI_API_KEY missing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import verifiers as vf

from curie_rlm_env.baseline_eval import (
    RETRIEVAL_TASKS,
    _attach_judge_to_env,
    _make_gemini_judge,
    _make_qwen_client,
    _run_one_rollout,
)
from curie_rlm_env.judge_cache import clear_cache

PROBE_TASKS: tuple[str, ...] = ("HFE", "DFT-S", "BIOGR")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 5b empirical pre-flight rollout-timing probe."
    )
    p.add_argument("--endpoint", required=True,
                   help="Qwen vLLM OpenAI-compatible base URL")
    p.add_argument("--output", default="results/probe_timings.json",
                   help="Output JSON path")
    p.add_argument("--n-rollouts", type=int, default=5,
                   help="Rollouts per task (default 5)")
    p.add_argument("--model", default="Qwen/Qwen3.5-7B-Instruct")
    return p.parse_args()


async def _probe_task(task_id: str, n_rollouts: int, client, judge, model: str) -> dict:
    env = vf.load_environment("curie-rlm-env", task_id=task_id, split="train")
    if task_id in RETRIEVAL_TASKS:
        if judge is None:
            raise RuntimeError(f"Task {task_id} requires Gemini judge but none built")
        _attach_judge_to_env(env, judge)
    semaphore = asyncio.Semaphore(2)
    sampling_args = {"temperature": 0.0, "max_tokens": 2048}
    n = min(n_rollouts, len(env.dataset))

    timings: list[dict] = []
    for i in range(n):
        clear_cache()  # per-rollout cache reset → measure raw cost (no caching benefit)
        t0 = time.time()
        result = await _run_one_rollout(env, i, client, model, sampling_args, semaphore)
        elapsed = time.time() - t0
        timings.append({
            "record_id": result.get("record_id", f"unknown_{i}"),
            "elapsed_seconds": round(elapsed, 3),
            "reward": result.get("reward", 0.0),
            "completion_length": result.get("completion_length", 0),
            "num_turns": result.get("num_turns", 0),
            "tool_call_count": result.get("tool_call_count", 0),
            "error": result.get("error"),
        })
        print(f"  [{task_id}] rollout {i+1}/{n}: {elapsed:.1f}s reward={result.get('reward', 0.0):.3f}", flush=True)

    elapsed_only = [t["elapsed_seconds"] for t in timings if t.get("error") is None]
    return {
        "task_id": task_id,
        "family": "retrieval" if task_id in RETRIEVAL_TASKS else (
            "geometric" if task_id == "BIOGR" else "freeform"
        ),
        "n_rollouts": len(timings),
        "n_succeeded": len(elapsed_only),
        "mean_seconds_per_rollout": (
            statistics.mean(elapsed_only) if elapsed_only else 0.0
        ),
        "max_seconds": max(elapsed_only) if elapsed_only else 0.0,
        "min_seconds": min(elapsed_only) if elapsed_only else 0.0,
        "rollouts": timings,
    }


def _recommendation(probe_results: list[dict]) -> dict:
    """Compute rollouts_per_example + step time recommendations from probe data."""
    by_task = {r["task_id"]: r for r in probe_results}

    target_step_minutes = 5.0  # design target — tune later
    batch_size = 64

    def _rpe(mean_s: float, lower_bound: int, upper_bound: int) -> int:
        if mean_s <= 0:
            return upper_bound
        # rollouts_per_example such that step_minutes ≈ target
        # step_seconds ≈ batch_size * rpe * mean_s / parallelism (approx 8 par)
        per_step = (target_step_minutes * 60 * 8) / (batch_size * mean_s)
        return max(lower_bound, min(upper_bound, int(per_step)))

    retrieval = by_task.get("DFT-S")
    freeform = by_task.get("HFE")
    geometric = by_task.get("BIOGR")

    rpe_retrieval = _rpe(retrieval["mean_seconds_per_rollout"], 2, 8) if retrieval else 4
    rpe_freeform = _rpe(freeform["mean_seconds_per_rollout"], 4, 16) if freeform else 16
    rpe_geometric = _rpe(geometric["mean_seconds_per_rollout"], 4, 16) if geometric else 16

    est_step_minutes_freeform = (
        (batch_size * rpe_freeform * freeform["mean_seconds_per_rollout"] / 8) / 60
        if freeform else 0.0
    )

    return {
        "rollouts_per_example_retrieval": rpe_retrieval,
        "rollouts_per_example_freeform": rpe_freeform,
        "rollouts_per_example_geometric": rpe_geometric,
        "estimated_step_minutes_freeform_phase": round(est_step_minutes_freeform, 2),
        "target_step_minutes": target_step_minutes,
        "batch_size_assumed": batch_size,
    }


async def _main_async(args: argparse.Namespace) -> int:
    # Hard-fail at startup (ZERO-FALLBACK)
    client = _make_qwen_client(args.endpoint)
    judge = _make_gemini_judge() if any(t in RETRIEVAL_TASKS for t in PROBE_TASKS) else None

    print(f"Probing {len(PROBE_TASKS)} tasks × {args.n_rollouts} rollouts each...")
    results: list[dict] = []
    for task_id in PROBE_TASKS:
        print(f"\nTask: {task_id}")
        r = await _probe_task(task_id, args.n_rollouts, client, judge, args.model)
        results.append(r)

    rec = _recommendation(results)
    output = {
        "model": args.model,
        "endpoint": args.endpoint,
        "n_rollouts_per_task": args.n_rollouts,
        "tasks": results,
        "recommendations": rec,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print()
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print(f"Recommended rollouts_per_example for retrieval: {rec['rollouts_per_example_retrieval']}")
    print(f"Recommended rollouts_per_example for free-form: {rec['rollouts_per_example_freeform']}")
    print(f"Recommended rollouts_per_example for geometric: {rec['rollouts_per_example_geometric']}")
    print(f"Estimated wall-clock per training step (batch_size={rec['batch_size_assumed']}): "
          f"{rec['estimated_step_minutes_freeform_phase']} minutes (free-form phase)")
    print()
    print(f"Wrote {args.output}")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
