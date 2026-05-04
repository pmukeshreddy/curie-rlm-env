"""Stage 3b — Curie scorers, verbatim from data/curie/colabs/curie_run_eval.ipynb.

Each function below mirrors the Curie eval notebook math byte-for-byte where
practical; deviations (NaN→0 in LLMSim, FASTA-only PDB extraction) are
documented inline and in CLAUDE.md "Documented Deviations from Curie".
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable

import json5
import numpy as np
from Bio import Align
from rouge_score import rouge_scorer
from bert_score import score as _bert_score_compute
import Levenshtein


# --- ROUGE-L ----------------------------------------------------------------
# Verbatim from data/curie/colabs/curie_run_eval.ipynb cell 22.

def _prepare_summary_rouge(summary: str) -> str:
    """Verbatim Curie cell 22: split sentences for rougeLsum correctness."""
    summary = summary.replace(" . ", " .\n")
    return summary


def rouge_l(pred: str, ref: str) -> dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-Lsum F-measure × 100. Verbatim Curie cell 22."""
    score_keys = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(score_keys)
    target = _prepare_summary_rouge(ref)
    prediction = _prepare_summary_rouge(pred)
    scores = scorer.score(target=target, prediction=prediction)
    return {key: scores[key].fmeasure * 100 for key in score_keys}


# --- BERTScore --------------------------------------------------------------
# Verbatim from data/curie/colabs/curie_run_eval.ipynb cell 20.

def bert_score_fn(pred: str, ref: str) -> dict[str, float]:
    """BERTScore precision/recall/F1 with lang='en'. Verbatim Curie cell 20."""
    precision, recall, F1 = _bert_score_compute(
        [pred], [ref], lang="en", verbose=False
    )
    return {
        "bert_precision": precision.item(),
        "bert_recall": recall.item(),
        "bert_f1": F1.item(),
    }


# --- IoU --------------------------------------------------------------------
# Verbatim from data/curie/colabs/curie_run_eval.ipynb cell 24
# (bb_intersection_over_union). Box layout: [W, S, E, N].

def iou(box_a, box_b) -> float:
    """IoU between two axis-aligned boxes. Verbatim Curie cell 24.

    Box layout per Curie cell 24 coords_to_box: [W, S, E, N].
    """
    box_a = np.asarray(box_a, dtype=float)
    box_b = np.asarray(box_b, dtype=float)

    def _intersection_area(box_a, box_b):
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        width = x_b - x_a
        height = y_b - y_a
        if (width < 0) or (height < 0):
            return 0.0
        return width * height

    def _area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    inter_area = _intersection_area(box_a, box_b)
    union_area = _area(box_a) + _area(box_b) - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / float(union_area)


# --- ID_r (PDB) -------------------------------------------------------------
# Verbatim from data/curie/colabs/curie_run_eval.ipynb cell 26
# (best_sequence_alignment_counts). Note: Curie's pdb_execute_code_eval branch
# is dropped per Stage 3b sandbox-safety decision; sequence extraction
# (FASTA `>` path) lives in rubric.py, not here.

def id_r(pred_seq: str, gt_seq: str) -> dict[str, Any]:
    """Pairwise alignment ID_r + counts. Verbatim Curie cell 26."""
    sequence_1 = pred_seq if pred_seq else " "
    sequence_2 = gt_seq if gt_seq else " "
    aligner = Align.PairwiseAligner()
    best_alignment = aligner.align(sequence_1, sequence_2)[0]

    max_length = max(len(sequence_1), len(sequence_2))
    if max_length == 0:
        normalized_distance: Any = "Zero length sequences"
    else:
        normalized_distance = (
            Levenshtein.distance(sequence_1, sequence_2) / max_length
        )
    if not best_alignment[0]:
        identity_ratio: Any = "Zero length alignment"
    else:
        identity_ratio = (
            best_alignment.counts().identities / len(best_alignment[0])
        )

    return {
        "n_gaps": best_alignment.counts().gaps,
        "n_identities": best_alignment.counts().identities,
        "n_mismatches": best_alignment.counts().mismatches,
        "normalized_levenshtein_distance": normalized_distance,
        "identity_ratio": identity_ratio,
    }


# --- LLMSim -----------------------------------------------------------------
# Adapted from data/curie/colabs/curie_run_eval.ipynb cells 14 + 18 (verbatim
# parsing + match counting). Empty-input NaN replaced with 0.0 for numeric
# reward stability — documented deviation.

def llm_sim(
    json_pred: list,
    json_ref: list,
    prompt_path: str,
    judge_client: Callable[[str], str],
) -> dict[str, float]:
    """LLMSim score: precision/recall/F1 from per-GT-item match calls.

    judge_client: callable taking a prompt str and returning the LLM response text.
    Returns dict with keys {precision, recall, f1, num_match, num_gt, num_response}.
    """
    template = Path(prompt_path).read_text()

    from .judge_cache import cached_llmsim_sync
    eval_list: list[dict[str, Any]] = []
    for j, gt_item in enumerate(json_ref):
        # Verbatim Curie cell 14: index injection to suppress hallucinated indices.
        for k, pred_item in enumerate(json_pred):
            if isinstance(pred_item, dict):
                pred_item["json_extracted_index"] = k
        prompt = (
            template
            .replace("{{json_ground_truth}}", json5.dumps(gt_item, indent=2))
            .replace("{{json_extracted_list}}", json5.dumps(json_pred, indent=2))
        )
        # Stage 5b: cached_llmsim_sync wraps judge_client with (gt, pred) keyed
        # cache. Cache lifetime = one training step; trainer calls clear_cache().
        # Outside training (Stage 3 tests), cache is harmless transparent layer.
        output = cached_llmsim_sync(judge_client, gt_item, json_pred, prompt)
        try:
            output_json = json5.loads(output)
        except Exception:
            inds = [m.start() for m in re.finditer(r",\s*\{", output)]
            if inds:
                output_json = json5.loads(output[: inds[-1]] + "]")
            else:
                output_json = []
        if isinstance(output_json, list):
            output_json = output_json[0] if output_json else {}
        eval_list.append(output_json)

    # Verbatim Curie cell 18 (eval_overall_result) with NaN → 0.0 substitution.
    num_match = sum(1 for item in eval_list if "json_extracted_index" in item)
    num_gt = len(json_ref)
    num_response = len(json_pred)
    pre = min(num_match / num_response, 1.0) if num_response else 0.0
    rec = min(num_match / num_gt, 1.0) if num_gt else 0.0
    f1 = 2.0 * pre * rec / (pre + rec) if (pre + rec) else 0.0
    if math.isnan(f1):
        f1 = 0.0
    # Curie cell 18 returns NaN on empty pred/ref (division by zero).
    # GRPO needs numeric reward; empty input → score 0.0 is the correct semantic.
    # This is input validation, not error masking — distinct from ZERO-FALLBACK rule.
    return {
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "num_match": num_match,
        "num_gt": num_gt,
        "num_response": num_response,
    }
