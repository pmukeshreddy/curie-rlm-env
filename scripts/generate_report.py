"""Stage 7 — generate markdown report from full_eval + rlm_ablation JSONs.

Internal results only. NO hardcoded numbers in the template body — all numbers
come from the JSON inputs via interpolation. Constants for thresholds live at
module top (SMALL_N_THRESHOLD, DELTA_HIGHLIGHT_THRESHOLD).

Stage 7 hotfix (ZERO-FALLBACK):
- Mandatory numeric fields are accessed via require(), which hard-fails with
  the full nested path on missing keys. No silent default values.
- The `std` field is optional (legitimately absent for n=1 tasks per
  statistics.stdev semantics). Rendered as "n/a" when missing — explicit
  schema-optional, NOT a fallback.

Usage:
    uv run python scripts/generate_report.py \\
        --eval results/full_eval.json \\
        --ablation results/rlm_ablation.json \\
        --output docs/REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Design thresholds (NOT result numbers — these are knobs in the report logic)
SMALL_N_THRESHOLD = 5
DELTA_HIGHLIGHT_THRESHOLD = 0.05

# Canonical ordered task list (mirrors curie_rlm_env.baseline_eval.TASK_IDS)
TASK_ORDER = (
    "DFT-S", "DFT-P", "DFT-C", "MPVE",
    "BIOGR", "PDB",
    "HFE", "HFD", "QECC_65", "GEO",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def require(d: Any, *path: Any) -> Any:
    """Strict nested-dict access. Raises KeyError with full path on miss.

    ZERO-FALLBACK: every mandatory eval-JSON field flows through this helper.
    """
    cur = d
    trail: list[str] = []
    for k in path:
        trail.append(str(k))
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(
                f"generate_report: missing required key in eval JSON: "
                f"'{'.'.join(trail)}'. Eval JSON is malformed or incomplete. "
                f"Inspect the source JSON and re-run the eval that produced it."
            )
        cur = cur[k]
    return cur


def _render_summary(metadata: dict, checkpoints: list[str]) -> str:
    # Metadata fields are display strings (not numerics) — '?' placeholder is
    # acceptable here per ZERO-FALLBACK (the rule covers numeric pulls).
    return (
        "## Summary\n\n"
        f"Internal eval of {len(checkpoints)} checkpoint(s) "
        f"({', '.join(checkpoints)}) on the CURIE benchmark "
        f"(split={metadata.get('split', '?')}, "
        f"n={metadata.get('test_split_n', '?')} records). "
        f"Policy model: {metadata.get('model', '?')}. "
        f"Judge model: {metadata.get('judge_model', '?')}. "
        f"Splits seed: {metadata.get('splits_seed', '?')}. "
        f"Timestamp: {metadata.get('timestamp', '?')}.\n"
    )


def _render_per_task_table(
    eval_data: dict[str, Any],
    checkpoints: list[str],
) -> str:
    lines = ["## Per-task results across checkpoints\n"]
    header = "| Task | " + " | ".join(checkpoints) + " |"
    sep = "|" + "|".join(["---"] * (len(checkpoints) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for task in TASK_ORDER:
        cells = [task]
        for ckpt in checkpoints:
            # Each require() carries the full path from eval_data root for clear KeyError messages.
            mean = require(eval_data, "per_checkpoint", ckpt, "per_task", task, "mean_reward")
            n = require(eval_data, "per_checkpoint", ckpt, "per_task", task, "n")
            # std is OPTIONAL: legitimately absent on n=1 tasks (statistics.stdev
            # raises on len<2). Render "n/a" when missing — schema-optional,
            # NOT a fallback.
            stats = require(eval_data, "per_checkpoint", ckpt, "per_task", task)
            std_val = stats.get("std")
            std_str = "n/a" if std_val is None else f"{std_val:.3f}"
            cells.append(f"{mean:.3f} ± {std_str} (n={n})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _render_delta_section(
    eval_data: dict[str, Any],
    checkpoints: list[str],
) -> str:
    if len(checkpoints) < 2:
        return "## Per-task delta\n\n(Need at least two checkpoints; skipped.)\n"
    first = checkpoints[0]
    last = checkpoints[-1]
    deltas: list[tuple[str, float, int]] = []
    for task in TASK_ORDER:
        first_mean = require(eval_data, "per_checkpoint", first, "per_task", task, "mean_reward")
        last_mean = require(eval_data, "per_checkpoint", last, "per_task", task, "mean_reward")
        n = require(eval_data, "per_checkpoint", last, "per_task", task, "n")
        deltas.append((task, last_mean - first_mean, n))
    deltas.sort(key=lambda x: -abs(x[1]))

    lines = [
        f"## Per-task delta: {last} vs {first}\n",
        f"Sorted by absolute improvement. Bold rows: |delta| >= {DELTA_HIGHLIGHT_THRESHOLD}. "
        f"Tasks with n < {SMALL_N_THRESHOLD} are flagged (small-N caveat applies).\n",
        "| Task | Delta | n | Notes |",
        "|---|---|---|---|",
    ]
    for task, d, n in deltas:
        flag_small = "small-N" if n < SMALL_N_THRESHOLD else ""
        bold = "**" if abs(d) >= DELTA_HIGHLIGHT_THRESHOLD else ""
        lines.append(f"| {bold}{task}{bold} | {bold}{d:+.3f}{bold} | {n} | {flag_small} |")
    lines.append("")
    return "\n".join(lines)


def _render_ablation_section(ablation: dict[str, Any]) -> str:
    if not ablation:
        return "## RLM ablation\n\n(No ablation JSON provided; skipped.)\n"
    on_per_task = require(ablation, "mode_A_rlm_on", "per_task")
    off_per_task = require(ablation, "mode_B_rlm_off", "per_task")
    delta_map = require(ablation, "delta_per_task")
    checkpoint = ablation.get("checkpoint", "?")  # display string

    lines = [
        f"## RLM ablation: scaffold on vs off (checkpoint = {checkpoint})\n",
        "| Task | RLM ON | RLM OFF | Delta |",
        "|---|---|---|---|",
    ]
    for task in TASK_ORDER:
        on_mean = require(ablation, "mode_A_rlm_on", "per_task", task, "mean_reward")
        off_mean = require(ablation, "mode_B_rlm_off", "per_task", task, "mean_reward")
        d = require(ablation, "delta_per_task", task)
        lines.append(f"| {task} | {on_mean:.3f} | {off_mean:.3f} | {d:+.3f} |")
    lines.append("")
    return "\n".join(lines)


def _render_caveats(eval_data: dict[str, Any], checkpoints: list[str]) -> str:
    # Compute small-N task list dynamically from the latest checkpoint
    small_n: list[str] = []
    if checkpoints:
        last = checkpoints[-1]
        for task in TASK_ORDER:
            n = require(eval_data, "per_checkpoint", last, "per_task", task, "n")
            if n < SMALL_N_THRESHOLD:
                small_n.append(f"{task} (n={n})")
    small_n_str = ", ".join(small_n) if small_n else "(none)"

    return (
        "## Honest caveats\n\n"
        f"- Test-split sizes per task: tasks below n={SMALL_N_THRESHOLD} flagged: {small_n_str}.\n"
        "- Judge model deviation: we use gemini-2.5-pro; Curie's released eval used gemini-1.5-pro-latest.\n"
        "- PDB scorer: code-exec branch dropped (sandbox safety). FASTA `>` extraction path only.\n"
        "- Free-form: BERTScore (Curie's released `_SHARED_METRCS`) replaces paper-only LMScore.\n"
        "- Sub-LM tokens not gradient-trained (standard action-masking, per Agent-R1 / prime-rl convention; "
        "see Stage 6.5 memo).\n"
    )


def _render_config_snapshot() -> str:
    yamls = ("safeguards.yaml", "judge.yaml", "rubric_dispatcher.yaml", "curie_tasks.yaml")
    tomls = ("curie_grpo_freeform.toml", "curie_grpo_retrieval.toml", "curie_grpo_geometric.toml")
    out = ["## Configuration snapshot\n"]
    for name in yamls:
        path = CONFIG_DIR / name
        if path.is_file():
            out.append(f"### config/{name}\n")
            out.append("```yaml")
            out.append(path.read_text().rstrip())
            out.append("```\n")
    for name in tomls:
        path = PROJECT_ROOT / "configs" / name
        if path.is_file():
            out.append(f"### configs/{name}\n")
            out.append("```toml")
            out.append(path.read_text().rstrip())
            out.append("```\n")
    return "\n".join(out)


def render_report(
    eval_data: dict[str, Any],
    ablation_data: dict[str, Any] | None,
) -> str:
    metadata = require(eval_data, "metadata")
    checkpoints = require(eval_data, "checkpoints")
    require(eval_data, "per_checkpoint")  # validate top-level structure exists

    parts = [
        "# Curie + RLM + DPPO+KL — Internal Results\n",
        _render_summary(metadata, checkpoints),
        _render_per_task_table(eval_data, checkpoints),
        _render_delta_section(eval_data, checkpoints),
        _render_ablation_section(ablation_data or {}),
        _render_caveats(eval_data, checkpoints),
        _render_config_snapshot(),
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", required=True, help="Path to results/full_eval.json")
    p.add_argument("--ablation", default=None, help="Path to results/rlm_ablation.json (optional)")
    p.add_argument("--output", required=True, help="Output markdown path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    eval_data = json.loads(Path(args.eval).read_text())
    ablation_data = json.loads(Path(args.ablation).read_text()) if args.ablation else None
    report = render_report(eval_data, ablation_data)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
