"""Stage 3b — Curie scorers, verbatim from data/curie/colabs/curie_run_eval.ipynb.

Each function below mirrors the Curie eval notebook math byte-for-byte where
practical; deviations (NaN→0 in LLMSim, FASTA-only PDB extraction) are
documented inline and in CLAUDE.md "Documented Deviations from Curie".
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import json5
import numpy as np
from Bio import Align
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import Levenshtein


# --- Free-form geometric coupling (CLAUDE.md guard #7) ----------------------
# (ROUGE_Lsum/100) ** alpha * BERT_F1 ** (1 - alpha). alpha biased toward
# ROUGE-L (the un-hackable metric); a zero on either component collapses the
# reward, closing the length-grift pathway in the prior additive 0.5/0.5 split.
FREEFORM_ROUGE_EXPONENT = 0.6

# IEEE-754 boundary slop, not a defensive wrap: BERTScore's identity-match
# F1 can come back as ~1 + 1.2e-7 (verified empirically against roberta-large)
# because cosine-similarity tensor arithmetic doesn't quite hit 1.0 exactly.
# Mathematical bound is still [0, 1]; we widen the *check* by this much so
# float noise at the boundary doesn't crash perfect-match rollouts. The value
# is passed through unchanged — no clipping, no normalization.
_FREEFORM_BOUND_SLOP = 1e-6


def freeform_geometric(rouge_lsum_norm: float, bert_f1: float) -> float:
    """Geometric coupling of normalized ROUGE-Lsum and BERTScore F1.

    Inputs must already lie in [0, 1]: the rubric divides ROUGE-Lsum by 100
    before calling. Zero on either component returns 0.0 — that signal must
    propagate intact, not be epsilon-smoothed. Inputs outside [0, 1] (beyond
    IEEE-754 boundary slop) raise.
    """
    lo = 0.0 - _FREEFORM_BOUND_SLOP
    hi = 1.0 + _FREEFORM_BOUND_SLOP
    for name, val in (("rouge_lsum_norm", rouge_lsum_norm), ("bert_f1", bert_f1)):
        if not (lo <= val <= hi):
            raise ValueError(f"freeform_geometric: {name}={val!r} not in [0, 1]")
    if rouge_lsum_norm == 0.0 or bert_f1 == 0.0:
        return 0.0
    alpha = FREEFORM_ROUGE_EXPONENT
    return (rouge_lsum_norm ** alpha) * (bert_f1 ** (1.0 - alpha))


# --- ROUGE-L ----------------------------------------------------------------
# Verbatim from data/curie/colabs/curie_run_eval.ipynb cell 22.

def _prepare_summary_rouge(summary: str) -> str:
    """Verbatim Curie cell 22: split sentences for rougeLsum correctness."""
    summary = summary.replace(" . ", " .\n")
    return summary


_ROUGE_SCORE_KEYS = ("rouge1", "rouge2", "rougeLsum")
_ROUGE_SCORER = rouge_scorer.RougeScorer(list(_ROUGE_SCORE_KEYS))


# lru_cache dedupes the rubric's twin call sites — _freeform_geometric_reward
# (headline) and _aux_rouge_lsum (weight-0 monitor) hit the same (pred, ref)
# per rollout, so without caching every rollout pays 2× the work.
# maxsize=128 caps RAM at ~1-2 MB and is well above the ~32 rollouts/step.
# Module-level _ROUGE_SCORER avoids reconstructing the tokenizer/stopword
# state on every cache miss.
@lru_cache(maxsize=128)
def rouge_l(pred: str, ref: str) -> dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-Lsum F-measure × 100. Verbatim Curie cell 22."""
    target = _prepare_summary_rouge(ref)
    prediction = _prepare_summary_rouge(pred)
    scores = _ROUGE_SCORER.score(target=target, prediction=prediction)
    return {key: scores[key].fmeasure * 100 for key in _ROUGE_SCORE_KEYS}


# --- BERTScore --------------------------------------------------------------
# Adapted from data/curie/colabs/curie_run_eval.ipynb cell 20. Documented
# Deviation (Stage 5): rescale_with_baseline=True replaces the Curie default
# of False — see CLAUDE.md "Documented Deviations from Curie release" and the
# pre-approved trigger at CLAUDE.md L54-59 (baseline-calibrated BERTScore as
# the anti-length-grift calibration fix). Rescaling subtracts the random-pair
# baseline and divides by (1 - baseline), so identity matches still score ~1.0
# but random-noise inputs score ~0 (raw was ~0.85), and worse-than-baseline
# inputs go negative — see the F1 clamp below.
#
# BERTScorer singleton: the previous `bert_score.score(...)` call re-loaded
# roberta-large (~1.4 GB) on every invocation — the orchestrator log showed
# 11× "RobertaModel LOAD REPORT" in a single batch, ~128 s of pure load
# overhead per training step. Holding the scorer resident eliminates that.
# device='cpu' is explicit: the orchestrator process owns 0 GPUs under
# prime-rl's deployment partitioning, so this stays off the saturated trainer
# (GPU 1) and vLLM (GPU 0) cards.

_BERT_SCORER: BERTScorer | None = None


def _get_bert_scorer() -> BERTScorer:
    global _BERT_SCORER
    if _BERT_SCORER is None:
        _BERT_SCORER = BERTScorer(
            lang="en", rescale_with_baseline=True, device="cpu"
        )
    return _BERT_SCORER


# Same dedupe as rouge_l: _freeform_geometric_reward (headline) and _aux_bert_f1
# (monitor) call this twice per rollout with identical args. Caching halves
# the encoder forward passes per step. maxsize=128 covers ~32 rollouts/step
# with headroom; entries are tiny (3 floats each).
@lru_cache(maxsize=128)
def bert_score_fn(pred: str, ref: str) -> dict[str, float]:
    """BERTScore precision/recall/F1 with lang='en', rescale_with_baseline=True.

    F1 is clamped to 0.0 at the scorer boundary: rescaled F1 goes negative when
    the prediction scores below the random-pair baseline (empirically ~-0.30
    for `'lorem ipsum xyz'` vs scientific GT, ~-0.48 for token-repetition
    hacks). The downstream geometric-mean combiner needs inputs in [0, 1] —
    propagating a negative would either raise via the range check or invert the
    reward sign. Clamping to 0 lets the zero-guard collapse the reward, which
    is the correct semantic: a below-baseline output gets no free-form credit.
    Precision/recall are returned raw (debugging signal only, no consumer).
    """
    precision, recall, F1 = _get_bert_scorer().score([pred], [ref])
    return {
        "bert_precision": precision.item(),
        "bert_recall": recall.item(),
        "bert_f1": max(0.0, F1.item()),
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
        except (ValueError, TypeError) as exc:
            # Strict: malformed judge output is a real error, not something to
            # repair into a valid list. Surface a truncated-safe excerpt so the
            # operator can diagnose without leaking arbitrary judge content.
            safe_excerpt = (output[:200] if isinstance(output, str) else repr(output))[:200]
            raise ValueError(
                f"LLMSim judge returned malformed JSON for gt index {j} "
                f"(raw output excerpt, ≤200 chars): {safe_excerpt!r}"
            ) from exc
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
