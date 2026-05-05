"""Stage 7 — retention and forgetting eval for continual CURIE-RLM.

Repository quotes anchoring this script:
- src/curie_rlm_env/continual.py: "Phase 2: 70% retrieval current tasks + 30% Phase 1 replay."
- src/curie_rlm_env/continual.py: "Phase 3: 60% geometric/structural current tasks + 20% Phase 1 replay"
- scripts/run_full_eval.py: "--checkpoints \"baseline,continual_phase1,continual_phase2,continual_phase3\""

Usage:
    uv run python scripts/eval_retention_forgetting.py \\
        --eval results/full_eval.json \\
        --output results/retention_forgetting.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from curie_rlm_env.continual import FREEFORM_TASKS, GEOMETRIC_TASKS, RETRIEVAL_TASKS

PHASE_CHECKPOINTS = {
    1: "continual_phase1",
    2: "continual_phase2",
    3: "continual_phase3",
}

TASKS_BY_CONTINUAL_PHASE = {
    1: FREEFORM_TASKS,
    2: RETRIEVAL_TASKS,
    3: GEOMETRIC_TASKS,
}


def require(d: Any, *path: Any) -> Any:
    """Strict nested-dict access with full missing-key path."""
    cur = d
    trail: list[str] = []
    for key in path:
        trail.append(str(key))
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(
                "eval_retention_forgetting: missing required key: "
                f"{'.'.join(trail)}"
            )
        cur = cur[key]
    return cur


def _mean_reward(eval_data: dict[str, Any], checkpoint: str, task_id: str) -> float:
    return float(require(eval_data, "per_checkpoint", checkpoint, "per_task", task_id, "mean_reward"))


def _retention_ratio(current_score: float, reference_score: float) -> float | None:
    if reference_score == 0.0:
        return None
    return current_score / reference_score


def _task_retention_record(
    eval_data: dict[str, Any],
    task_id: str,
    learned_continual_phase: int,
    target_checkpoint: str,
) -> dict[str, Any]:
    learned_checkpoint = PHASE_CHECKPOINTS[learned_continual_phase]
    learned_score = _mean_reward(eval_data, learned_checkpoint, task_id)
    target_score = _mean_reward(eval_data, target_checkpoint, task_id)
    return {
        "task_id": task_id,
        "learned_continual_phase": learned_continual_phase,
        "learned_checkpoint": learned_checkpoint,
        "target_checkpoint": target_checkpoint,
        "learned_score": learned_score,
        "target_score": target_score,
        "forgetting": learned_score - target_score,
        "retention_delta": target_score - learned_score,
        "retention_ratio": _retention_ratio(target_score, learned_score),
    }


def _phase_summary(
    records: list[dict[str, Any]],
    source_continual_phase: int,
    target_checkpoint: str,
) -> dict[str, Any]:
    forgetting = [float(record["forgetting"]) for record in records]
    retention_delta = [float(record["retention_delta"]) for record in records]
    ratios = [
        float(record["retention_ratio"])
        for record in records
        if record["retention_ratio"] is not None
    ]
    return {
        "source_continual_phase": source_continual_phase,
        "target_checkpoint": target_checkpoint,
        "n_tasks": len(records),
        "mean_forgetting": statistics.mean(forgetting),
        "mean_retention_delta": statistics.mean(retention_delta),
        "mean_retention_ratio": statistics.mean(ratios) if ratios else None,
    }


def compute_retention_forgetting(
    eval_data: dict[str, Any],
    final_checkpoint: str = "continual_phase3",
) -> dict[str, Any]:
    """Compute per-task forgetting from first-learned checkpoint to target checkpoints."""
    checkpoints = require(eval_data, "checkpoints")
    for checkpoint in PHASE_CHECKPOINTS.values():
        if checkpoint not in checkpoints:
            raise KeyError(f"eval_retention_forgetting: checkpoint missing from checkpoints: {checkpoint}")
    if final_checkpoint not in checkpoints:
        raise KeyError(f"eval_retention_forgetting: final checkpoint missing from checkpoints: {final_checkpoint}")

    per_task_final: dict[str, dict[str, Any]] = {}
    for continual_phase, task_ids in TASKS_BY_CONTINUAL_PHASE.items():
        for task_id in task_ids:
            per_task_final[task_id] = _task_retention_record(
                eval_data,
                task_id,
                continual_phase,
                final_checkpoint,
            )

    phase_summaries: list[dict[str, Any]] = []
    for source_continual_phase, target_checkpoint in (
        (1, "continual_phase2"),
        (1, "continual_phase3"),
        (2, "continual_phase3"),
    ):
        records = [
            _task_retention_record(eval_data, task_id, source_continual_phase, target_checkpoint)
            for task_id in TASKS_BY_CONTINUAL_PHASE[source_continual_phase]
        ]
        phase_summaries.append(
            _phase_summary(records, source_continual_phase, target_checkpoint)
        )

    return {
        "final_checkpoint": final_checkpoint,
        "per_task_final": per_task_final,
        "phase_summaries": phase_summaries,
    }


def write_retention_output(result: dict[str, Any], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", required=True, help="Path to results/full_eval.json")
    p.add_argument("--output", required=True, help="Output retention/forgetting JSON path")
    p.add_argument("--final-checkpoint", default="continual_phase3")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    eval_data = json.loads(Path(args.eval).read_text())
    result = compute_retention_forgetting(
        eval_data,
        final_checkpoint=args.final_checkpoint,
    )
    write_retention_output(result, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
