"""Single-rollout-per-task debug probe for `has_final_answer=False`.

Runs ONE rollout per CURIE task (10 total) against the local Qwen vLLM
endpoint, captures the per-task `[CURIE-DEBUG]` stderr log to its own file,
and prints a summary table showing whether each rollout produced a final
answer and whether `submit_answer` was ever invoked.

Per CLAUDE.md ZERO-FALLBACK: hard-fails if --endpoint is missing. For
retrieval tasks (DFT-S, DFT-P, MPVE), if GEMINI_API_KEY is unset the rubric
scoring will throw — that's caught per-task and recorded as `rubric_error`,
because the `answer_schema_valid` debug log we care about fires during the
rollout (before scoring).

Usage:
    uv run python scripts/debug_has_answer_flow.py --endpoint $QWEN_ENDPOINT

Output:
    results/debug/<task_id>.stderr.log  — per-task [CURIE-DEBUG] output
    stdout                               — summary table
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import verifiers as vf

from curie_rlm_env.baseline_eval import (
    RETRIEVAL_TASKS,
    TASK_IDS,
    _attach_judge_to_env,
    _make_qwen_client,
)
from curie_rlm_env.judge import make_gemini_judge_from_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Debug probe: one rollout per task, capture has_final_answer signals."
    )
    p.add_argument("--endpoint", required=True,
                   help="Qwen vLLM OpenAI-compatible base URL (REQUIRED, ZERO-FALLBACK)")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out-dir", default="results/debug",
                   help="Per-task stderr log directory (default results/debug)")
    return p.parse_args()


async def _run_one_task(
    task_id: str,
    split: str,
    client: Any,
    model: str,
    judge: Any,
    log_path: Path,
) -> dict[str, Any]:
    """Run ONE rollout for `task_id` and return a summary dict.

    Stderr (where [CURIE-DEBUG] writes) is redirected to `log_path` for the
    duration of the rollout. Exceptions during rollout/scoring are caught and
    recorded; we do NOT propagate them so the loop can continue across all
    10 tasks.
    """
    summary: dict[str, Any] = {
        "task_id": task_id,
        "has_final_answer": None,
        "submit_answer_calls": None,
        "n_turns": None,
        "reward": None,
        "rollout_error": None,
        "rubric_error_in_state": None,
        "log_path": str(log_path),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_fh:
        with contextlib.redirect_stderr(log_fh):
            try:
                env = vf.load_environment("curie-rlm-env", task_id=task_id, split=split)
                if task_id in RETRIEVAL_TASKS and judge is not None:
                    _attach_judge_to_env(env, judge)

                example_idx = 0
                info = env.dataset[example_idx].get("info", {}) or {}
                input_dict = {
                    "prompt": env.dataset[example_idx]["prompt"],
                    "answer": env.dataset[example_idx].get("answer", ""),
                    "task": env.task_id,
                    "info": info,
                    "example_id": example_idx,
                }
                output = await env.run_rollout(
                    input=input_dict,
                    client=client,
                    model=model,
                    sampling_args={"temperature": 0.0, "max_tokens": 2048},
                )
                state = getattr(output, "state", None) or {}
                summary["has_final_answer"] = "final_answer" in state
                summary["submit_answer_calls"] = state.get("_curie_submit_answer_calls", 0)
                summary["n_turns"] = len(state.get("trajectory") or [])
                summary["reward"] = state.get("reward")
                summary["rubric_error_in_state"] = state.get("error")
            except Exception as exc:  # noqa: BLE001 — debug script: capture everything
                summary["rollout_error"] = f"{type(exc).__name__}: {exc}"
                # Emit traceback to the per-task log so we have full context.
                traceback.print_exc(file=sys.stderr)

    return summary


def _print_summary(rows: list[dict[str, Any]]) -> None:
    headers = [
        "task_id", "has_final", "submit_calls", "n_turns",
        "reward", "rollout_error", "rubric_error", "log",
    ]
    widths = {
        "task_id": 10, "has_final": 10, "submit_calls": 13, "n_turns": 8,
        "reward": 8, "rollout_error": 40, "rubric_error": 30, "log": 38,
    }
    print()
    print("=" * 160)
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-" * 160)
    for r in rows:
        row_vals = {
            "task_id": str(r["task_id"]),
            "has_final": str(r["has_final_answer"]),
            "submit_calls": str(r["submit_answer_calls"]),
            "n_turns": str(r["n_turns"]),
            "reward": (f"{r['reward']:.3f}" if isinstance(r["reward"], (int, float)) else str(r["reward"])),
            "rollout_error": (str(r["rollout_error"])[: widths["rollout_error"]] if r["rollout_error"] else "-"),
            "rubric_error": (str(r["rubric_error_in_state"])[: widths["rubric_error"]] if r["rubric_error_in_state"] else "-"),
            "log": str(r["log_path"])[-widths["log"]:],
        }
        print(" | ".join(row_vals[h].ljust(widths[h]) for h in headers))
    print("=" * 160)
    n_with_answer = sum(1 for r in rows if r["has_final_answer"] is True)
    n_with_submit = sum(1 for r in rows if (r["submit_answer_calls"] or 0) > 0)
    print(f"\n{n_with_answer}/{len(rows)} tasks produced has_final_answer=True")
    print(f"{n_with_submit}/{len(rows)} tasks invoked submit_answer at least once")
    print(f"\nPer-task stderr logs written to: {Path(rows[0]['log_path']).parent if rows else '(no rows)'}")


async def _main_async(args: argparse.Namespace) -> int:
    client = _make_qwen_client(args.endpoint)
    # Try to build the Gemini judge for retrieval tasks. If it fails (no key,
    # missing dep), continue without it — those tasks will record `rubric_error`
    # but the answer-presence debug log still fires during the rollout.
    judge = None
    try:
        judge = make_gemini_judge_from_env()
        print(f"[debug-probe] Gemini judge built; will attach to retrieval tasks: {sorted(RETRIEVAL_TASKS)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[debug-probe] Gemini judge unavailable ({type(exc).__name__}: {exc}); "
              f"retrieval tasks will likely show rubric_error", file=sys.stderr)

    out_dir = Path(args.out_dir)
    rows: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        log_path = out_dir / f"{task_id}.stderr.log"
        print(f"[debug-probe] running {task_id} → {log_path}", flush=True)
        row = await _run_one_task(task_id, args.split, client, args.model, judge, log_path)
        rows.append(row)
        print(
            f"  done: has_final={row['has_final_answer']} "
            f"submit_calls={row['submit_answer_calls']} "
            f"n_turns={row['n_turns']} "
            f"err={row['rollout_error'] or '-'}",
            flush=True,
        )

    _print_summary(rows)
    return 0


def main() -> int:
    args = parse_args()
    if not args.endpoint:
        print("ERROR: --endpoint is required (ZERO-FALLBACK)", file=sys.stderr)
        return 2
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
