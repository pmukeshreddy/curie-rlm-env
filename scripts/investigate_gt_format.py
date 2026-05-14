"""Stage 3b follow-up: is `json.dumps(entry["ground_truth"])` corrupting BERTScore?

After reverting `rescale_with_baseline=True` we have a working reward signal,
but BERT_F1 raw clusters in [0.70, 0.80] across all 16 baseline rollouts —
basically a constant. All reward variance now comes from ROUGE-L. That's
suspicious for a metric that's supposed to be the semantic-match signal.

Hypothesis: the rubric's `answer` field comes from
`json.dumps(entry["ground_truth"])` (datasets.py:148), which for a string GT
adds enclosing quotes and converts every newline to a literal `\\n` escape
sequence. BERTScore then compares the model's clean prose against an
escaped-quote string, which compresses the F1 distribution toward the
random-pair baseline (around 0.72 on roberta-large for English).

Test: load one free-form GT directly from disk for each of the five free-form
tasks. Compute BERT_F1 between a representative model prediction and:
  (a) the GT in the form datasets.py serves it (json.dumps on the json5-loaded value)
  (b) the GT as a "raw" Python string (best-effort extraction from whatever shape it is on disk)

If (b) is significantly higher than (a) for most tasks → json-encoding is the
real root cause and the right fix is to stop json-encoding free-form GTs
inside datasets.py (LLMSim/IoU/IDr tasks still need it; only free-form is
plain-text scored). After that fix we could re-enable `rescale_with_baseline`
since legitimate predictions would no longer be in the negative tail.

If (b) ≈ (a) → json-encoding is NOT the issue; baseline Qwen3-8B genuinely
produces below-baseline text for these tasks (which is consistent with the
post-revert clustering at ~0.72-0.80) and the rescale revert was the right
permanent fix.

Usage:
    INFERENCE_SERVER_IP unused (no model calls).
    uv run python scripts/investigate_gt_format.py
"""
from __future__ import annotations

import json
from pathlib import Path

import json5

from curie_rlm_env.datasets import TASK_MAP
from curie_rlm_env.scorers import bert_score_fn

# Five free-form tasks from continual.py: FREEFORM_TASKS
FREEFORM_TASKS = ["DFT-C", "HFE", "HFD", "QECC_65", "GEO"]

# Representative baseline-quality predictions per task. Hand-written generic
# scientific prose — short summaries on the rough topic, the kind of output
# any vanilla 7B-class instruct model produces when asked to summarize a paper
# in a domain it's never specifically trained on. NOT meant to be the right
# answer for any specific record_id; the point is to test whether json-encoding
# the GT changes the BERT score on a fixed prediction.
PREDICTIONS = {
    "DFT-C": (
        "The paper presents density functional theory calculations on phosphorene "
        "as an anode material. Key findings include strong magnesium binding, "
        "low diffusion barriers around 0.09 eV, and a theoretical specific "
        "capacity near 865 mAh/g at 0.833 V. Bulk black phosphorus shows similar "
        "advantages but suffers from 33% volumetric expansion."
    ),
    "HFE": (
        "The study investigates heterogeneous catalysis on metal surfaces using "
        "first-principles DFT. Adsorption energies, transition states, and "
        "reaction barriers are computed for the proposed reaction pathway. "
        "The activation energy is reduced compared to homogeneous catalysis."
    ),
    "HFD": (
        "The paper describes the structural and electronic properties of a novel "
        "two-dimensional material. Band structure calculations reveal a direct "
        "bandgap, and phonon analysis confirms dynamical stability. The material "
        "shows promise for optoelectronic applications."
    ),
    "QECC_65": (
        "The paper introduces permutation-invariant quantum codes constructed "
        "from real polynomials with multiple roots at roots of unity. These codes "
        "correct t errors and require at least (2t+1)^2(d-1) qudits for "
        "dimension d. The construction leverages symmetric group actions."
    ),
    "GEO": (
        "The study analyzes spatial distribution patterns using remote sensing "
        "data and machine learning. The proposed model achieves higher accuracy "
        "than baseline approaches and generalizes across different geographic "
        "regions. Applications include land-cover mapping and environmental "
        "monitoring."
    ),
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data" / "curie" / "data" / "data"


def load_first_gt(task_id: str) -> tuple[str, object]:
    """Return (record_id, raw_gt) for the first GT file of a task."""
    folder, _ = TASK_MAP[task_id]
    gt_dir = _DATA_ROOT / folder / "ground_truth"
    gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        raise FileNotFoundError(f"No GT files in {gt_dir}")
    gt_file = gt_files[0]
    record_id = gt_file.stem
    raw_gt = json5.loads(gt_file.read_text())
    return record_id, raw_gt


def to_raw_string(raw_gt: object) -> str:
    """Best-effort: return a "natural" string form of a GT object.

    For free-form tasks the GT is typically already a string. For dict-shaped
    GTs we concatenate the string-typed values in field order so BERT sees
    real prose, not JSON syntax. This is the kind of thing a fixed
    `_row_from_split_entry` would do for free-form rows.
    """
    if isinstance(raw_gt, str):
        return raw_gt
    if isinstance(raw_gt, dict):
        parts = []
        for k, v in raw_gt.items():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                parts.extend(str(x) for x in v if isinstance(x, (str, int, float)))
        return "\n".join(parts) if parts else json.dumps(raw_gt)
    if isinstance(raw_gt, list):
        return "\n".join(
            x if isinstance(x, str) else json.dumps(x) for x in raw_gt
        )
    return json.dumps(raw_gt)


def main() -> int:
    print("=== GT-format investigation ===\n")
    print("For each free-form task, compare BERT_F1 of a fixed prediction against")
    print("  (a) GT served by datasets.py: json.dumps(json5.loads(gt_file))")
    print("  (b) GT in raw string form (concatenated field values for dict GTs)\n")

    rows = []
    for task_id in FREEFORM_TASKS:
        try:
            record_id, raw_gt = load_first_gt(task_id)
        except FileNotFoundError as exc:
            print(f"[{task_id}] SKIP: {exc}")
            continue

        gt_dumps = json.dumps(raw_gt)
        gt_raw = to_raw_string(raw_gt)
        pred = PREDICTIONS[task_id]

        bert_dumps = bert_score_fn(pred, gt_dumps)["bert_f1"]
        bert_raw = bert_score_fn(pred, gt_raw)["bert_f1"]
        delta = bert_raw - bert_dumps

        rows.append((task_id, record_id, raw_gt, gt_dumps, gt_raw, bert_dumps, bert_raw, delta))

        print(f"[{task_id}] record={record_id}")
        print(f"  raw_gt type:        {type(raw_gt).__name__}")
        if isinstance(raw_gt, dict):
            print(f"  raw_gt dict keys:   {sorted(raw_gt.keys())}")
        print(f"  json.dumps len:     {len(gt_dumps):>7}    head: {gt_dumps[:120]!r}")
        print(f"  raw_string len:     {len(gt_raw):>7}    head: {gt_raw[:120]!r}")
        print(f"  pred len:           {len(pred):>7}    head: {pred[:120]!r}")
        print(f"  BERT_F1 vs json.dumps:  {bert_dumps:.4f}")
        print(f"  BERT_F1 vs raw_string:  {bert_raw:.4f}")
        print(f"  delta (raw - dumps):    {delta:+.4f}")
        print()

    print("\n=== SUMMARY ===")
    print(f"{'task':<10s} {'record_id':<28s} {'BERT(dumps)':>11s} {'BERT(raw)':>11s} {'delta':>8s}")
    for task_id, record_id, _, _, _, bd, br, d in rows:
        print(f"{task_id:<10s} {record_id:<28s} {bd:>11.4f} {br:>11.4f} {d:>+8.4f}")

    if not rows:
        print("(no rows; data files missing)")
        return 1

    deltas = [r[7] for r in rows]
    mean_delta = sum(deltas) / len(deltas)
    n_positive = sum(1 for d in deltas if d > 0.02)
    n_negative_or_flat = sum(1 for d in deltas if d <= 0.02)

    print(f"\nMean delta (raw - dumps):   {mean_delta:+.4f}")
    print(f"  tasks with delta > +0.02:   {n_positive}/{len(deltas)}  (raw GT scores meaningfully higher)")
    print(f"  tasks with delta <= +0.02:  {n_negative_or_flat}/{len(deltas)}  (json.dumps does NOT meaningfully hurt)")

    print("\n--- VERDICT ---")
    if n_positive >= 3:
        print("JSON-ENCODING IS A REAL ROOT CAUSE.")
        print("  Most tasks score meaningfully higher with raw-string GT than with json.dumps GT.")
        print("  Fix: in src/curie_rlm_env/datasets.py:_row_from_split_entry, stop json.dumps-ing")
        print("  the GT for free-form tasks (DFT-C, HFE, HFD, QECC_65, GEO). Retrieval/geometric/")
        print("  structural tasks (DFT-S, DFT-P, MPVE, BIOGR, PDB) keep json.dumps because the")
        print("  rubric needs json5.loads on those answers (CurieRubric._loads_ref).")
        print("  After that fix, re-evaluating rescale_with_baseline=True is a sensible follow-up:")
        print("  legitimate predictions may then land in positive territory.")
    else:
        print("JSON-ENCODING IS NOT THE ROOT CAUSE.")
        print("  Raw-string and json.dumps GTs score essentially the same.")
        print("  Baseline Qwen3-8B genuinely produces near-baseline text for these tasks.")
        print("  The rescale revert was the right permanent fix; no follow-up needed on")
        print("  the GT format. Length-grift watch in CLAUDE.md stays in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
