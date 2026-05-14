"""Stage 3b — Curie scorers, verbatim from data/curie/colabs/curie_run_eval.ipynb.

Each function below mirrors the Curie eval notebook math byte-for-byte where
practical; deviations (NaN→0 in LLMSim, FASTA-only PDB extraction) are
documented inline and in CLAUDE.md "Documented Deviations from Curie".
"""
from __future__ import annotations

import math
import re
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

# IEEE-754 boundary slop for BERT input ONLY: BERTScore's identity-match F1
# can come back as ~1 + 1.2e-7 (verified empirically against roberta-large)
# because cosine-similarity tensor arithmetic doesn't quite hit 1.0 exactly.
# ROUGE-Lsum has no such slop — it's computed by rouge_score and divided by
# 100.0, strict [0, 1] by construction. Applying slop there would absorb real
# upstream bugs (e.g., a forgotten /100) silently.
_FREEFORM_BOUND_SLOP = 1e-6


def freeform_geometric(rouge_lsum_norm: float, bert_f1: float) -> float:
    """Geometric coupling of normalized ROUGE-Lsum and BERTScore F1.

    ROUGE is validated STRICTLY against [0, 1] (no slop — out-of-range
    indicates an upstream bug, surface it). BERT is validated with IEEE-754
    boundary slop on either side, then snapped to [0, 1] before the geometric
    formula so the output is strictly bounded by 1.0. Inputs must already lie
    in [0, 1] by contract; the rubric divides ROUGE-Lsum by 100 and clamps
    negative rescaled BERT to 0 at the consumption site before calling here.
    Zero on either (post-snap) component returns 0.0 — the zero signal must
    propagate, not be epsilon-smoothed.
    """
    if not (0.0 <= rouge_lsum_norm <= 1.0):
        raise ValueError(
            f"freeform_geometric: rouge_lsum_norm={rouge_lsum_norm!r} not in [0, 1] (strict)"
        )
    if not (0.0 - _FREEFORM_BOUND_SLOP <= bert_f1 <= 1.0 + _FREEFORM_BOUND_SLOP):
        raise ValueError(
            f"freeform_geometric: bert_f1={bert_f1!r} not in [0, 1] (±IEEE slop)"
        )
    bert_f1 = min(1.0, max(0.0, bert_f1))
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

# Stage 5 Documented Deviation from Curie cell 22: ROUGE-L is computed with
# stopword filtering. rouge_score's DefaultTokenizer has no `stopwords` param,
# so we plug in a custom tokenizer that wraps the library's tokenize.tokenize
# call and drops common English function words AFTER stemming. Triggered by
# the empirical content-free baseline (stopword-only string scored ~13/100 on
# scientific GT) interacting with the geometric coupling — independent reward
# even when the model produces nothing of substance. Filtering drops the floor
# to <5/100. List = NLTK english stopwords embedded inline (no nltk dep).
from rouge_score import tokenize as _rouge_tokenize

_ROUGE_STOPWORDS: frozenset[str] = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now",
})


class _RougeStopwordTokenizer:
    """rouge_score Tokenizer interface implementation that drops English
    stopwords AFTER the library's stemming step (stemming runs only on tokens
    >3 chars, so short function words like 'the'/'of' aren't transformed).

    Stopword list intentionally generic — NO scientific-domain extensions.
    Removing 'experiment', 'study', 'method', 'result' would penalize
    legitimate scientific writing.
    """

    def __init__(self, stemmer):
        self._stemmer = stemmer
        self._stopwords = _ROUGE_STOPWORDS

    def tokenize(self, text):
        tokens = _rouge_tokenize.tokenize(text, self._stemmer)
        return [t for t in tokens if t not in self._stopwords]


# rouge_score's DefaultTokenizer uses NLTK's PorterStemmer when
# use_stemmer=True. We construct the same stemmer once and share it with the
# custom tokenizer (matches the library's default pattern).
from nltk.stem.porter import PorterStemmer  # noqa: E402

_ROUGE_TOKENIZER = _RougeStopwordTokenizer(PorterStemmer())
_ROUGE_SCORER = rouge_scorer.RougeScorer(
    list(_ROUGE_SCORE_KEYS), tokenizer=_ROUGE_TOKENIZER,
)


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
# Verbatim Curie cell 20: BERTScorer(lang="en") with library defaults. The
# previous Stage 5 deviation flipped rescale_with_baseline to True (CLAUDE.md
# L54-59 pre-approval) on the assumption that legitimate predictions would
# score positive on the rescaled scale. The Stage 3b ZMQ harness on Phase 1
# data refuted that assumption: 16/16 baseline Qwen3-8B rollouts produced
# rescaled BERT_F1 in [-0.77, -0.16] — uniformly below the random English
# baseline. That zero-clamp at the consumption site collapsed the geometric
# coupling for every rollout, DAPO online_difficulty_filtering rejected every
# group, and the trainer was stuck at step 0. CLAUDE.md L62 ("defenses are
# added with W&B evidence in Stage 5+, never preemptively") is the project
# rule that mandates this revert: rescaling was a preemptive anti-length-grift
# defense, the empirical evidence from the harness shows it kills the reward
# signal, so we revert to the Curie default. Length-grift is a Stage 5 watch
# item and will be addressed with a different mechanism (e.g. an output-length
# cap inside _freeform_geometric_reward) once we have W&B evidence of it.
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
        _BERT_SCORER = BERTScorer(lang="en", device="cpu")
    return _BERT_SCORER


# Same dedupe as rouge_l: _freeform_geometric_reward (headline) and _aux_bert_f1
# (monitor) call this twice per rollout with identical args. Caching halves
# the encoder forward passes per step. maxsize=128 covers ~32 rollouts/step
# with headroom; entries are tiny (3 floats each).
@lru_cache(maxsize=128)
def bert_score_fn(pred: str, ref: str) -> dict[str, float]:
    """BERTScore precision/recall/F1 with lang='en' (Curie cell 20 verbatim).

    F1 is in [0, 1] with a high English-pair floor (~0.85 for any English text
    against scientific GT). The geometric coupling in `freeform_geometric` and
    its rubric clamp tolerate any value in that range; with raw BERTScore the
    clamp is a no-op and the geometric mean stays well-defined for every
    rollout, restoring DAPO advantage variance at step 0 of GRPO training.
    """
    precision, recall, F1 = _get_bert_scorer().score([pred], [ref])
    return {
        "bert_precision": precision.item(),
        "bert_recall": recall.item(),
        "bert_f1": F1.item(),
    }


# --- DIoU (BIOGR) -----------------------------------------------------------
# Stage 5 Documented Deviation from Curie cell 24:
#   1. Plain IoU replaced with Distance-IoU (Zheng et al. 2020):
#        DIoU = IoU - ρ²(centers) / c²(enclosing diagonal)
#      DIoU is in [-1, 1]; non-overlapping bboxes get a NEGATIVE gradient
#      signal proportional to center distance — fixes the IoU sparse-gradient
#      cliff (plain IoU returns exactly 0 for any non-overlap regardless of
#      how close, giving RL nothing to climb).
#   2. Antimeridian handling: a bbox with W > E (e.g. W=170, E=-170) is split
#      into [W, 180] and [-180, E] before intersection/union; centers are
#      computed in a "shifted" coord system so the bbox is contiguous.
#   3. Reward clamp NOT here — clamped at rubric.py consumption site
#      (same pattern as bert_score_fn / rescaled BERT). Raw negative output
#      stays available for the _aux_diou_raw observability metric.
#
# Validation is strict — S>=N, |lat|>90, zero-area bboxes raise. These are
# reference-data or prediction-format bugs, not silent failures.

def _split_at_antimeridian(box) -> list[tuple[float, float, float, float]]:
    """[W, S, E, N] → 1 or 2 non-crossing sub-bboxes. W==E (zero width) caller-validated."""
    W, S, E, N = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if W <= E:
        return [(W, S, E, N)]
    return [(W, S, 180.0, N), (-180.0, S, E, N)]


def _bbox_area(box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _pair_intersection_area(box_a, box_b) -> float:
    W = max(box_a[0], box_b[0])
    S = max(box_a[1], box_b[1])
    E = min(box_a[2], box_b[2])
    N = min(box_a[3], box_b[3])
    if E <= W or N <= S:
        return 0.0
    return (E - W) * (N - S)


def _shifted_center(box) -> tuple[float, float]:
    """Center longitude in the bbox's natural (possibly wrapping) frame.

    For a non-crossing bbox, midpoint of [W, E]. For a crossing bbox (W > E),
    treat the eastward span as [W, E + 360] and take that midpoint. Returned
    longitude may lie outside [-180, 180]; the diou() caller wraps differences
    to handle the shorter great-circle separation.
    """
    W, S, E, N = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if W <= E:
        cx = (W + E) / 2.0
    else:
        cx = (W + E + 360.0) / 2.0
    cy = (S + N) / 2.0
    return cx, cy


def _validate_bbox(box, name: str) -> None:
    W, S, E, N = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if S >= N:
        raise ValueError(f"{name}: S>=N (latitude not strictly increasing): S={S}, N={N}")
    if W == E:
        raise ValueError(f"{name}: zero-width bbox (W==E={W})")
    if S == N:
        raise ValueError(f"{name}: zero-height bbox (S==N={S})")
    if S < -90.0 or N > 90.0:
        raise ValueError(f"{name}: latitude out of [-90, 90]: S={S}, N={N}")


def diou(box_a, box_b) -> float:
    """Distance-IoU on (W, S, E, N) lat/lon bboxes, antimeridian-aware.

    Returns the raw DIoU in [-1, 1]. Non-overlapping bboxes return NEGATIVE
    values proportional to center distance — that's the desired gradient.
    The rubric layer clamps to [0, 1] for the reward; the aux observability
    metric (_aux_diou_raw) keeps the raw value for Stage 5 W&B.
    """
    _validate_bbox(box_a, "box_a")
    _validate_bbox(box_b, "box_b")

    a_subs = _split_at_antimeridian(box_a)
    b_subs = _split_at_antimeridian(box_b)

    inter = sum(_pair_intersection_area(a, b) for a in a_subs for b in b_subs)
    area_a = sum(_bbox_area(s) for s in a_subs)
    area_b = sum(_bbox_area(s) for s in b_subs)
    union = area_a + area_b - inter
    iou_score = inter / union if union > 0 else 0.0

    cx_a, cy_a = _shifted_center(box_a)
    cx_b, cy_b = _shifted_center(box_b)
    dx = cx_a - cx_b
    # Wrap to the shorter direction around the sphere.
    if dx > 180.0:
        dx -= 360.0
    elif dx < -180.0:
        dx += 360.0
    rho_sq = dx * dx + (cy_a - cy_b) ** 2

    # Enclosing-bbox diagonal: pick the union of all sub-bbox extents in
    # standard 2D space. For antimeridian-crossing bboxes this overstates the
    # diagonal slightly (no spherical correction); acceptable given the
    # ρ²/c² ratio is bounded and the metric stays in [-1, 1] in practice.
    all_subs = a_subs + b_subs
    enc_W = min(s[0] for s in all_subs)
    enc_S = min(s[1] for s in all_subs)
    enc_E = max(s[2] for s in all_subs)
    enc_N = max(s[3] for s in all_subs)
    c_sq = (enc_E - enc_W) ** 2 + (enc_N - enc_S) ** 2
    if c_sq == 0:
        return iou_score
    return iou_score - rho_sq / c_sq


# --- ID_r (PDB) -------------------------------------------------------------
# Adapted from data/curie/colabs/curie_run_eval.ipynb cell 26
# (best_sequence_alignment_counts). Curie's pdb_execute_code_eval branch is
# dropped per Stage 3b sandbox-safety decision; sequence extraction
# (FASTA `>` path) lives in rubric.py, not here.
#
# Stage 5 Documented Deviation from cell 26:
#   1. Length floor (30 absolute, 0.3 fraction of ref) — rejects sub-domain
#      stubs ("M" universal start codon, etc.) before alignment.
#   2. Explicit denom max(len_pred, len_ref) — removes the alignment-length-
#      with-internal-gaps variation; aligns with the "did the model
#      reproduce the full sequence" question.
#   3. Strict-identity scoring (match=1, mismatch=0, gap=-1) — these are
#      Biopython PairwiseAligner() defaults but we set them explicitly so a
#      future Biopython version change can't silently shift to BLOSUM credit.
#      Note: BLOSUM is the biologically-correct similarity metric, but for
#      RL reward we want a sharp residue-exact match signal, not gradient
#      credit for conservative substitutions (L↔I, V↔I, etc.).

_PDB_LENGTH_FLOOR_ABS = 30  # smallest plausible functional protein domain
_PDB_LENGTH_FLOOR_FRAC = 0.3  # 30% of ref length


def id_r(pred_seq: str, gt_seq: str) -> dict[str, Any]:
    """Length-normalized identity ratio with hard length floor.

    Returns:
        identity_ratio: matched / max(len_pred, len_ref); 0.0 if floor rejects.
        matched: raw exact-identity count from global alignment.
        len_pred, len_ref: sequence lengths (diagnostics).
        length_floor_rejected: bool, True iff floor triggered (Stage 5 W&B).
        n_gaps, n_mismatches: Curie cell 26 counts (preserved for compat).
    """
    len_pred = len(pred_seq) if pred_seq else 0
    len_ref = len(gt_seq) if gt_seq else 0

    floor = max(_PDB_LENGTH_FLOOR_ABS, int(_PDB_LENGTH_FLOOR_FRAC * len_ref))
    if len_pred < floor:
        return {
            "identity_ratio": 0.0,
            "matched": 0,
            "len_pred": len_pred,
            "len_ref": len_ref,
            "length_floor_rejected": True,
            "n_gaps": 0,
            "n_mismatches": 0,
        }

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -1
    best_alignment = aligner.align(pred_seq, gt_seq)[0]
    counts = best_alignment.counts()
    matched = counts.identities

    denom = max(len_pred, len_ref)
    identity_ratio = matched / denom if denom > 0 else 0.0

    return {
        "identity_ratio": identity_ratio,
        "matched": matched,
        "len_pred": len_pred,
        "len_ref": len_ref,
        "length_floor_rejected": False,
        "n_gaps": counts.gaps,
        "n_mismatches": counts.mismatches,
    }


# --- Numeric-value verifier (CLAUDE.md guard #2) ----------------------------
# Deterministic post-LLM filter on LLMSim's match decisions. Closes the
# within-record verbosity grift on numeric fields: Gemini might accept
# "2.5 eV (measured at 300K via photoluminescence, reported in Table 3)" as
# matching a GT bandgap of "2.1 eV" because the surrounding prose is plausible;
# the verifier revokes that match on a 20% magnitude disagreement (tolerance
# locked at 5% per CLAUDE.md guard #2 / config/safeguards.yaml:43). Strictly
# post-LLM, strictly a filter — can only revoke, never add matches.
#
# °C is intentionally excluded from temperature aliases: K↔°C is affine
# (offset 273.15), not multiplicative, and our converter is factor-only.
# In-scope unit families per the spec: energy, length, inverse-length, temperature.

_UNIT_CONVERSIONS: dict[str, tuple[str, float]] = {
    # energy → eV
    "eV": ("eV", 1.0),
    "meV": ("eV", 1e-3),
    "electron-volts": ("eV", 1.0),
    "electron-volt": ("eV", 1.0),
    "electronvolts": ("eV", 1.0),
    "J": ("J", 1.0),
    "kcal/mol": ("kcal/mol", 1.0),
    # length → Å
    "Å": ("Å", 1.0),
    "angstrom": ("Å", 1.0),
    "angstroms": ("Å", 1.0),
    "nm": ("Å", 10.0),
    "pm": ("Å", 0.01),
    # inverse-length → cm⁻¹
    "cm⁻¹": ("cm⁻¹", 1.0),
    "1/cm": ("cm⁻¹", 1.0),
    # temperature → K
    "K": ("K", 1.0),
    "Kelvin": ("K", 1.0),
    "kelvin": ("K", 1.0),
}

# Longest aliases first so "electron-volts" wins over "eV" and "kcal/mol" wins
# over "K". Right-boundary lookahead avoids partial matches inside larger words.
_NUM_UNIT_RE = re.compile(
    r"(?:~|≈|approximately\s+)?\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"\s*"
    r"(" + "|".join(re.escape(u) for u in sorted(_UNIT_CONVERSIONS, key=len, reverse=True)) + r")"
    r"(?=$|[\s,;.()\[\]])"
)


def _extract_numeric_values(record: dict) -> dict[str, tuple[float, str]]:
    """Per-record numeric extractor. Walks top-level string fields, returns
    {field: (canonical_magnitude, canonical_unit)} for each parseable numeric.

    Strategy: find the FIRST (magnitude, unit) match in the value via
    _NUM_UNIT_RE; convert to canonical via _UNIT_CONVERSIONS. Unparseable
    fields are omitted (not zero, not None — silently absent so the verifier
    knows it has no opinion on that field). Non-string values (nested dicts,
    ints, bools, lists) are skipped — text-field fuzziness is the LLM's job.
    """
    out: dict[str, tuple[float, str]] = {}
    if not isinstance(record, dict):
        return out
    for field, value in record.items():
        if not isinstance(value, str):
            continue
        m = _NUM_UNIT_RE.search(value)
        if not m:
            continue
        try:
            mag = float(m.group(1))
        except ValueError:
            continue
        unit_raw = m.group(2)
        canonical_unit, factor = _UNIT_CONVERSIONS[unit_raw]
        out[field] = (mag * factor, canonical_unit)
    return out


def _verify_numeric_match(
    ref_record: dict, pred_record: dict, tolerance: float = 0.05,
) -> bool:
    """Programmatic spot-check after LLMSim. Returns True iff every numeric
    field present in BOTH records agrees in canonical unit AND within `tolerance`
    relative error. If no field overlaps on parseable numerics, returns True
    (verifier abstains — that's "no opinion", not a fallback).

    Tolerance default 5% locked per CLAUDE.md guard #2.
    """
    ref_nums = _extract_numeric_values(ref_record)
    pred_nums = _extract_numeric_values(pred_record)
    overlap = ref_nums.keys() & pred_nums.keys()
    if not overlap:
        return True
    for field in overlap:
        ref_mag, ref_unit = ref_nums[field]
        pred_mag, pred_unit = pred_nums[field]
        if ref_unit != pred_unit:
            return False
        denom = max(abs(ref_mag), abs(pred_mag), 1e-12)
        if abs(ref_mag - pred_mag) / denom > tolerance:
            return False
    return True


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
            if not output_json:
                # Strict: empty-list judge output is outside the judge prompt's
                # contract (which yields a matched record or a non-list null).
                # Conflating "empty list" with "no match" silently scores it as
                # zero — surface it so a real prompt confusion is diagnosable.
                safe_excerpt = (output[:200] if isinstance(output, str) else repr(output))[:200]
                raise ValueError(
                    f"LLMSim judge returned empty JSON list for gt index {j} "
                    f"(raw output excerpt, ≤200 chars): {safe_excerpt!r}"
                )
            output_json = output_json[0]
        eval_list.append(output_json)

    # Programmatic numeric verifier (CLAUDE.md guard #2): post-filter on
    # Gemini's match decisions, revoking claimed matches where any overlapping
    # numeric field disagrees beyond 5% relative tolerance. Strictly post-LLM,
    # strictly a filter — can only revoke, never add matches. See
    # _verify_numeric_match docstring for abstain semantics.
    verifier_revoked_count = 0
    for j, item in enumerate(eval_list):
        if not isinstance(item, dict) or "json_extracted_index" not in item:
            continue
        pred_index = item["json_extracted_index"]
        if not isinstance(pred_index, int) or not (0 <= pred_index < len(json_pred)):
            continue
        ref_record = json_ref[j]
        pred_record = json_pred[pred_index]
        if not (isinstance(ref_record, dict) and isinstance(pred_record, dict)):
            continue
        if not _verify_numeric_match(ref_record, pred_record, tolerance=0.05):
            del item["json_extracted_index"]
            verifier_revoked_count += 1

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
        "verifier_revoked_count": verifier_revoked_count,
    }
