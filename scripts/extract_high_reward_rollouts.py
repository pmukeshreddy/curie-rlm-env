"""Stage 5.5 — extract high-reward rollouts for RFT bootstrap.

Reads prime-rl rollout records (schema-agnostic — prime-rl's exact rollout
output schema is NOT verified from web docs per Stage 5b OQ-A precedent).
Filters by reward threshold, caps per-task, writes prime-rl SFT format
(per Stage 4a §3) to a single JSONL file.

Output schema (per Stage 4a memo):
    {"messages": [...], "task_id": "...", "reward": 0.7}

ZERO-FALLBACK (default):
- Missing --rollouts-dir → hard fail.
- Empty filtered set (0 above threshold) → hard fail with remediation hint.
- Malformed JSON line OR record missing required fields → hard fail with the
  offending file/line excerpt.

To opt into the legacy skip-and-continue behavior, pass --allow-skip-malformed.
That flag is intended for one-off forensic runs, not the default path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _normalize_record(record: Any) -> dict[str, Any] | None:
    """Schema-agnostic field extraction. Returns None if any required field missing."""
    if not isinstance(record, dict):
        return None

    # messages: prefer top-level "messages"; else combine prompt+completion
    messages = record.get("messages")
    if messages is None:
        prompt = record.get("prompt")
        completion = record.get("completion")
        if not isinstance(prompt, list) or not isinstance(completion, list):
            return None
        messages = list(prompt) + list(completion)
    if not isinstance(messages, list) or not messages:
        return None

    # reward: try several common locations
    reward = record.get("reward")
    if reward is None:
        info_dict = record.get("info")
        if isinstance(info_dict, dict):
            reward = info_dict.get("reward")
    if reward is None:
        reward = record.get("score")
    if reward is None:
        return None
    try:
        reward_f = float(reward)
    except (TypeError, ValueError):
        return None

    # task_id: try info.task_id, then top-level task_id, then top-level task
    task_id = None
    info_dict = record.get("info")
    if isinstance(info_dict, dict):
        task_id = info_dict.get("task_id")
    if task_id is None:
        task_id = record.get("task_id") or record.get("task")
    if not isinstance(task_id, str) or not task_id:
        return None

    return {"messages": messages, "task_id": task_id, "reward": reward_f}


def _filter_and_cap(
    normalized: list[dict[str, Any]],
    threshold: float,
    max_per_task: int,
) -> list[dict[str, Any]]:
    """Filter by reward threshold, cap each task at max_per_task (top-K by reward)."""
    above = [r for r in normalized if r["reward"] >= threshold]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in above:
        by_task.setdefault(r["task_id"], []).append(r)
    capped: list[dict[str, Any]] = []
    for task_id, recs in by_task.items():
        # Stratified-by-task: keep top-K by reward (highest-quality demos first)
        recs_sorted = sorted(recs, key=lambda x: -x["reward"])
        capped.extend(recs_sorted[:max_per_task])
    return capped


def _print_histogram(rewards: list[float], n_buckets: int = 10) -> None:
    if not rewards:
        return
    lo, hi = min(rewards), max(rewards)
    span = hi - lo if hi > lo else 1.0
    bucket_size = span / n_buckets
    buckets = [0] * n_buckets
    for r in rewards:
        idx = int((r - lo) / bucket_size) if hi > lo else 0
        idx = min(idx, n_buckets - 1)
        buckets[idx] += 1
    max_count = max(buckets) or 1
    bar_w = 40
    print(
        f"\nReward distribution ({len(rewards)} rollouts, "
        f"range [{lo:.3f}, {hi:.3f}]):"
    )
    for i, count in enumerate(buckets):
        bucket_lo = lo + i * bucket_size
        bucket_hi = lo + (i + 1) * bucket_size
        bar = "#" * int(bar_w * count / max_count) if count else ""
        print(f"  [{bucket_lo:.2f}, {bucket_hi:.2f}): {bar} ({count})")


def _print_summary(
    raw_count: int,
    skipped: int,
    above_count: int,
    capped: list[dict[str, Any]],
    threshold: float,
) -> None:
    print(f"\nProcessed {raw_count} records ({skipped} skipped as malformed).")
    print(f"Above threshold {threshold}: {above_count}")
    print(f"After per-task cap: {len(capped)}")
    by_task: dict[str, int] = {}
    for r in capped:
        by_task[r["task_id"]] = by_task.get(r["task_id"], 0) + 1
    print("\nPer-task counts in output:")
    for task_id in sorted(by_task):
        print(f"  {task_id:10s}: {by_task[task_id]}")
    _print_histogram([r["reward"] for r in capped])


def extract_records(
    rollouts_dir: Path | str,
    output: Path | str,
    threshold: float,
    max_per_task: int,
    verbose: bool = True,
    allow_skip_malformed: bool = False,
) -> list[dict[str, Any]]:
    """Main extraction. Used by CLI and tests.

    By default, malformed records (bad JSON OR missing required fields) raise
    ValueError with the offending file/line. Pass `allow_skip_malformed=True`
    to opt into the legacy skip-and-continue behavior.
    """
    rollouts_path = Path(rollouts_dir)
    if not rollouts_path.is_dir():
        raise SystemExit(
            f"ERROR: rollouts directory does not exist: {rollouts_path}"
        )
    paths = sorted(rollouts_path.glob("*.jsonl"))
    if not paths:
        raise SystemExit(
            f"ERROR: no *.jsonl files found under {rollouts_path}"
        )

    raw: list[dict[str, Any]] = []
    skipped = 0
    raw_count = 0
    for p in paths:
        for lineno, line in enumerate(p.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            raw_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if allow_skip_malformed:
                    skipped += 1
                    continue
                excerpt = line[:200]
                raise ValueError(
                    f"Malformed JSON in {p}:{lineno}: {exc}; "
                    f"line excerpt (≤200 chars): {excerpt!r}. "
                    f"Pass --allow-skip-malformed to skip and continue."
                ) from exc
            norm = _normalize_record(record)
            if norm is None:
                if allow_skip_malformed:
                    skipped += 1
                    continue
                raise ValueError(
                    f"Record at {p}:{lineno} is missing required fields "
                    f"(messages|prompt+completion, reward|info.reward|score, task_id|info.task_id|task). "
                    f"Pass --allow-skip-malformed to skip and continue."
                )
            raw.append(norm)

    above = [r for r in raw if r["reward"] >= threshold]
    capped = _filter_and_cap(raw, threshold, max_per_task)

    if not capped:
        raise SystemExit(
            f"ERROR: RFT requires at least N high-reward rollouts; "
            f"got 0 above threshold {threshold}. "
            f"Lower threshold or run more Stage 5 steps before extracting."
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for r in capped:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    if verbose:
        _print_summary(raw_count, skipped, len(above), capped, threshold)
        print(f"\nWrote {len(capped)} records to {output_path}")

    return capped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts-dir", required=True, help="Directory containing rollout *.jsonl files")
    p.add_argument("--output", required=True, help="Output JSONL path for filtered+capped records")
    p.add_argument("--threshold", type=float, default=0.5, help="Reward threshold (records with reward >= T pass)")
    p.add_argument("--max-per-task", type=int, default=50, help="Per-task cap on retained records (top-K by reward)")
    p.add_argument(
        "--allow-skip-malformed",
        action="store_true",
        help="Opt-in: skip and count malformed records instead of failing. Default: strict.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    extract_records(
        rollouts_dir=args.rollouts_dir,
        output=args.output,
        threshold=args.threshold,
        max_per_task=args.max_per_task,
        allow_skip_malformed=args.allow_skip_malformed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
