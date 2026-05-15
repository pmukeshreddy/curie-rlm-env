"""Stage 3b — CurieRubric per-task scoring dispatcher.

Reward weights per CLAUDE.md L21-25 + config/rubric_dispatcher.yaml + Stage 3b
locked decisions:
- DFT-S, DFT-P, MPVE: LLMSim × 0.7  (max 0.7)
- BIOGR: IoU × 1.0
- PDB: ID_r × 1.0  (FASTA `>` path only; code-exec branch dropped)
- DFT-C, HFE, HFD, QECC_65, GEO: (ROUGE-Lsum/100)^0.6 * BERT-F1^0.4 × 1.0  (max 1.0)
  where BERT-F1 is the Curie cell 20 default (raw, no baseline rescale).

Auxiliary metrics (weight 0, applied to ALL 10 tasks for observability via
RubricGroup aggregate): rouge_lsum, bert_f1 (raw F1 — same scorer).

Stage 5 update (CLAUDE.md guard #7): free-form scoring is a geometric coupling,
not an additive split — zero on either component collapses the reward, closing
the length-grift pathway. BERT-F1 uses Curie cell 20 verbatim (raw, no
baseline rescale); the previous Stage 5 deviation (rescale_with_baseline=True)
was reverted after the Stage 3b ZMQ harness on Phase 1 data showed that the
rescaled F1 went uniformly negative for baseline Qwen3-8B rollouts and
collapsed every rollout's reward to 0 (DAPO online_difficulty_filtering then
rejected every group → trainer stuck at step 0). See CLAUDE.md "Documented
Deviations from Curie release" entry for the W&B evidence and the rule
("defenses are added with W&B evidence in Stage 5+, never preemptively").
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import json5
import verifiers as vf

from .scorers import bert_score_fn, diou, freeform_geometric, id_r, llm_sim, rouge_l


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FROZEN_PROMPTS = _PROJECT_ROOT / "config" / "frozen_prompts" / "llmsim"

_LLMSIM_PROMPT = {
    "DFT-S": _FROZEN_PROMPTS / "dft_structure.txt",
    "DFT-P": _FROZEN_PROMPTS / "dft_metadata.txt",
    "MPVE":  _FROZEN_PROMPTS / "mat.txt",
}

_RETRIEVAL_TASKS = frozenset({"DFT-S", "DFT-P", "MPVE"})
_FREEFORM_TASKS = frozenset({"DFT-C", "HFE", "HFD", "QECC_65", "GEO"})
_GEOMETRIC_TASKS = frozenset({"BIOGR"})
_STRUCTURAL_TASKS = frozenset({"PDB"})
_ALL_TASKS = (
    _RETRIEVAL_TASKS | _FREEFORM_TASKS | _GEOMETRIC_TASKS | _STRUCTURAL_TASKS
)


# Free-form GT preprocessing — verbatim port of `preprocess_ground_truth` from
# `data/curie/colabs/curie_run_eval.ipynb` cell 28 (else-branch, which is the
# code path all five free-form tasks DFT-C/HFE/HFD/QECC_65/GEO take):
#
#     json_gt = json5.loads(ground_truth)
#     # Drop only these three identifier fields (top-level for dict GTs,
#     # per-item for list GTs).
#     if isinstance(json_gt, dict):
#         json_gt.pop("record_id", None)
#         json_gt.pop("arxiv_id", None)
#         json_gt.pop("paper_id", None)
#     if isinstance(json_gt, list):
#         for item in json_gt:
#             if isinstance(item, dict):
#                 item.pop("record_id", None)
#                 item.pop("arxiv_id", None)
#                 item.pop("paper_id", None)
#     return str(json5.dumps(json_gt))
#
# Earlier versions of this module did per-task field extraction (HFE
# Hamiltonian+Other_info, HFD task+Note per step, DFT-C `code`, etc.) on the
# theory that Curie scored against extracted prose. That was a misreading: the
# Curie scorer passes the whole json-dumped GT (minus 3 identifier fields) to
# ROUGE/BERT for every free-form task. Per-task whitelists therefore drift
# from upstream and have been removed.
_CURIE_IDENTIFIER_FIELDS = ("record_id", "arxiv_id", "paper_id")


class CurieRubric(vf.Rubric):
    """Per-task reward dispatcher. See module docstring for full mapping."""

    def __init__(
        self,
        judge_client: Optional[Callable[[str], str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._judge_client = judge_client
        # Per-task headline reward funcs.
        self.add_reward_func(self._llmsim_reward, weight=0.7)
        self.add_reward_func(self._diou_reward, weight=1.0)
        self.add_reward_func(self._idr_reward, weight=1.0)
        self.add_reward_func(self._freeform_geometric_reward, weight=1.0)
        # Auxiliary observability metrics (weight 0 — applied to all tasks).
        self.add_metric(self._aux_rouge_lsum)
        self.add_metric(self._aux_bert_f1)
        self.add_metric(self._aux_diou_raw)

    @staticmethod
    def _extract_pred(completion, state) -> str:
        """Pull prediction text from state.final_answer (preferred) or completion."""
        if isinstance(state, dict) and state.get("final_answer"):
            return state["final_answer"]
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list) and completion:
            last = completion[-1]
            if isinstance(last, dict):
                return last.get("content", "") or ""
        return ""

    @staticmethod
    def _task_id(info, task) -> str:
        if isinstance(info, dict) and "task_id" in info:
            return info["task_id"]
        return task or ""

    @staticmethod
    def _safe_loads_pred(text: str):
        """Parse a *prediction* string. None == invalid prediction → caller returns 0."""
        if not text:
            return None
        try:
            return json5.loads(text)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _loads_ref(answer: str, task_id: str):
        """Parse a *reference* string. Strict: malformed reference raises ValueError."""
        if not isinstance(answer, str) or not answer:
            raise ValueError(
                f"CurieRubric[{task_id}]: reference answer is empty or not a string"
            )
        try:
            return json5.loads(answer)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"CurieRubric[{task_id}]: malformed reference JSON: {exc}"
            ) from exc

    @staticmethod
    def _freeform_reference(answer_string: str) -> str:
        """Curie cell 28 `preprocess_ground_truth` (else-branch), verbatim.

        For all five free-form tasks (DFT-C, HFE, HFD, QECC_65, GEO), Curie
        passes the whole json-dumped GT to ROUGE/BERT, with only three
        identifier fields stripped: record_id, arxiv_id, paper_id (top-level
        for dict GTs, per-item for list GTs). Returns `str(json5.dumps(...))`
        of the cleaned object.

        Falls back to the original `answer_string` when the GT can't be parsed
        as JSON. This preserves behavior on a malformed answer instead of
        zeroing the reward; it does NOT mask scoring bugs since the original
        answer would have been the input either way.
        """
        try:
            gt = json5.loads(answer_string)
        except (ValueError, TypeError):
            return answer_string
        if isinstance(gt, dict):
            for key in _CURIE_IDENTIFIER_FIELDS:
                gt.pop(key, None)
        elif isinstance(gt, list):
            for item in gt:
                if isinstance(item, dict):
                    for key in _CURIE_IDENTIFIER_FIELDS:
                        item.pop(key, None)
        return str(json5.dumps(gt))

    # ------- Headline reward funcs -----------------------------------------
    # Strict semantics:
    #   * Invalid prediction → return 0.0 silently (model's fault).
    #   * Invalid / missing reference → raise ValueError (data/infra fault).

    async def _llmsim_reward(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
        task_id = self._task_id(info, task)
        if task_id not in _RETRIEVAL_TASKS:
            return 0.0
        if self._judge_client is None:
            raise ValueError(f"LLMSim for {task_id} requires judge_client; got None")
        json_ref = self._loads_ref(answer, task_id)  # reference issues raise
        pred_text = self._extract_pred(completion, state)
        if not pred_text or not pred_text.strip():
            return 0.0
        json_pred = self._safe_loads_pred(pred_text)
        if json_pred is None:
            return 0.0
        if not isinstance(json_pred, list):
            json_pred = [json_pred]
        if not isinstance(json_ref, list):
            json_ref = [json_ref]
        prompt_path = str(_LLMSIM_PROMPT[task_id])
        result = llm_sim(json_pred, json_ref, prompt_path, self._judge_client)
        return float(result["f1"])

    async def _diou_reward(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
        task_id = self._task_id(info, task)
        if task_id not in _GEOMETRIC_TASKS:
            return 0.0
        ref_box = self._loads_ref(answer, task_id)
        if not isinstance(ref_box, dict):
            raise ValueError(
                f"CurieRubric[{task_id}]: reference must be a JSON object; got {type(ref_box).__name__}"
            )
        for key in ("W", "S", "E", "N"):
            if key not in ref_box:
                raise ValueError(
                    f"CurieRubric[{task_id}]: reference missing required field {key!r}"
                )

        pred_text = self._extract_pred(completion, state)
        if not pred_text.strip():
            return 0.0
        pred_box = self._safe_loads_pred(pred_text)
        if not isinstance(pred_box, dict):
            return 0.0
        try:
            raw_diou = diou(
                [pred_box["W"], pred_box["S"], pred_box["E"], pred_box["N"]],
                [ref_box["W"], ref_box["S"], ref_box["E"], ref_box["N"]],
            )
        except (KeyError, TypeError, ValueError):
            # Pred missing one of W/S/E/N, non-numeric, or fails diou's strict
            # bbox validation (S>=N, |lat|>90, zero-area) → invalid prediction → 0.
            return 0.0
        # DIoU is in [-1, 1] — negative values are the diagnostic "gradient
        # without overlap" signal that _aux_diou_raw logs. The reward must
        # stay in [0, 1]; clamp at the consumption site, same pattern as the
        # rescaled-BERT clamp in _freeform_geometric_reward.
        return max(0.0, float(raw_diou))

    async def _idr_reward(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
        task_id = self._task_id(info, task)
        if task_id not in _STRUCTURAL_TASKS:
            return 0.0
        ref_obj = self._loads_ref(answer, task_id)
        if not isinstance(ref_obj, dict):
            raise ValueError(
                f"CurieRubric[{task_id}]: reference must be a JSON object; got {type(ref_obj).__name__}"
            )
        if "sequence" not in ref_obj:
            raise ValueError(
                f"CurieRubric[{task_id}]: reference missing required field 'sequence'"
            )
        ref_seq = ref_obj["sequence"]
        if not isinstance(ref_seq, str) or not ref_seq:
            raise ValueError(
                f"CurieRubric[{task_id}]: reference 'sequence' must be a non-empty string"
            )

        pred_text = self._extract_pred(completion, state)
        if not pred_text.strip():
            return 0.0
        # FASTA `>` extraction only (Stage 3b sandbox safety: code-exec dropped).
        pred_seq = ""
        lines = pred_text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith(">") and i < len(lines) - 1:
                pred_seq = lines[i + 1].strip()
                break
        if not pred_seq:
            pred_seq = pred_text.strip()
        # id_r returns identity_ratio as float in [0, 1] under the Stage 5
        # length-floor + length-normalized rewrite. The old string-typed
        # "Zero length alignment" branch is removed — length_floor_rejected
        # now carries the same signal explicitly.
        result = id_r(pred_seq, ref_seq)
        return float(result["identity_ratio"])

    async def _freeform_geometric_reward(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
        task_id = self._task_id(info, task)
        if task_id not in _FREEFORM_TASKS:
            return 0.0
        if not isinstance(answer, str) or not answer:
            raise ValueError(
                f"CurieRubric[{task_id}]: reference answer is empty or not a string"
            )
        pred_text = self._extract_pred(completion, state)
        if not pred_text.strip():
            return 0.0
        # Curie-verbatim preprocessing (cell 28 else-branch): strip 3
        # identifier fields from the json-dumped GT before passing to
        # ROUGE/BERT. Matches how Curie scores all free-form tasks.
        ref_text = self._freeform_reference(answer)
        rouge_norm = rouge_l(pred_text, ref_text)["rougeLsum"] / 100.0
        # bert_score_fn returns raw BERTScore F1 (Curie cell 20 verbatim, no
        # baseline rescale) — strictly in [0, 1], with a high English floor so
        # the geometric coupling stays well-defined for every rollout. The clamp
        # below is a defensive bound (max(0, raw) is a no-op on the raw scale,
        # but kept as explicit input-domain enforcement for freeform_geometric).
        bert_f1_raw = bert_score_fn(pred_text, ref_text)["bert_f1"]
        bert_f1_for_geometric = max(0.0, bert_f1_raw)
        return float(freeform_geometric(rouge_norm, bert_f1_for_geometric))

    # ------- Auxiliary observability metrics (weight 0, all tasks) ---------

    async def _aux_rouge_lsum(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
        pred_text = self._extract_pred(completion, state)
        if not pred_text.strip() or not answer:
            return 0.0
        # Match the headline reward path: strip JSON syntax + identifier
        # metadata from free-form references before scoring. Non-free-form
        # tasks keep the json-encoded answer (LLMSim/IoU/IDr have their own
        # structured parsers; the aux metric here just observes for W&B).
        task_id = self._task_id(info, task)
        ref_text = (
            self._freeform_reference(answer)
            if task_id in _FREEFORM_TASKS
            else answer
        )
        return float(rouge_l(pred_text, ref_text)["rougeLsum"])

    async def _aux_bert_f1(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
        # Logs raw BERTScore F1 (Curie cell 20 verbatim — no baseline rescale).
        # F1 is in [0, 1] with a high English floor (~0.85). Used for Stage 5
        # W&B observability across all 10 tasks; the headline reward consumes
        # the same scorer in _freeform_geometric_reward, with the same per-task
        # reference extraction so headline and aux stay numerically consistent.
        pred_text = self._extract_pred(completion, state)
        if not pred_text.strip() or not answer:
            return 0.0
        task_id = self._task_id(info, task)
        ref_text = (
            self._freeform_reference(answer)
            if task_id in _FREEFORM_TASKS
            else answer
        )
        return float(bert_score_fn(pred_text, ref_text)["bert_f1"])

    async def _aux_diou_raw(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
        # Logs raw DIoU (can be negative for non-overlapping bboxes). The
        # headline _diou_reward clamps to [0, 1]; this aux keeps the negative
        # tail for Stage 5 W&B — distance signal stays visible even when the
        # clamp zeroes the reward. BIOGR only; non-bbox tasks return 0.0.
        task_id = self._task_id(info, task)
        if task_id not in _GEOMETRIC_TASKS:
            return 0.0
        if not isinstance(answer, str) or not answer:
            return 0.0
        try:
            ref_box = json5.loads(answer)
        except (ValueError, TypeError):
            return 0.0
        if not isinstance(ref_box, dict) or not all(k in ref_box for k in ("W", "S", "E", "N")):
            return 0.0
        pred_text = self._extract_pred(completion, state)
        if not pred_text.strip():
            return 0.0
        pred_box = self._safe_loads_pred(pred_text)
        if not isinstance(pred_box, dict):
            return 0.0
        try:
            return float(diou(
                [pred_box["W"], pred_box["S"], pred_box["E"], pred_box["N"]],
                [ref_box["W"], ref_box["S"], ref_box["E"], ref_box["N"]],
            ))
        except (KeyError, TypeError, ValueError):
            return 0.0
