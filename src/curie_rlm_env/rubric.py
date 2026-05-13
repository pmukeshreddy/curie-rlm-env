"""Stage 3b — CurieRubric per-task scoring dispatcher.

Reward weights per CLAUDE.md L21-25 + config/rubric_dispatcher.yaml + Stage 3b
locked decisions:
- DFT-S, DFT-P, MPVE: LLMSim × 0.7  (max 0.7)
- BIOGR: IoU × 1.0
- PDB: ID_r × 1.0  (FASTA `>` path only; code-exec branch dropped)
- DFT-C, HFE, HFD, QECC_65, GEO: (ROUGE-Lsum/100)^0.6 * BERT-F1^0.4 × 1.0  (max 1.0)
  where BERT-F1 is rescaled (rescale_with_baseline=True) and clamped to [0,1].

Auxiliary metrics (weight 0, applied to ALL 10 tasks for observability via
RubricGroup aggregate): rouge_lsum, bert_f1 (also rescaled+clamped — same
scorer).

Stage 5 update (CLAUDE.md guard #7): free-form scoring is a geometric coupling,
not an additive split — zero on either component collapses the reward, closing
the length-grift pathway. BERT-F1 is now baseline-rescaled per CLAUDE.md L54-59
(approved calibration fix), with the negative-tail clamped to 0 at the scorer
boundary so the geometric domain stays valid. The set of headline reward
FUNCTIONS is still minimal (no anti-hack guards bolted onto the rubric); the
change is to the free-form scoring formula only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import json5
import verifiers as vf

from .scorers import bert_score_fn, freeform_geometric, id_r, iou, llm_sim, rouge_l


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
        self.add_reward_func(self._iou_reward, weight=1.0)
        self.add_reward_func(self._idr_reward, weight=1.0)
        self.add_reward_func(self._freeform_geometric_reward, weight=1.0)
        # Auxiliary observability metrics (weight 0 — applied to all tasks).
        self.add_metric(self._aux_rouge_lsum)
        self.add_metric(self._aux_bert_f1)

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

    async def _iou_reward(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
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
            return float(iou(
                [pred_box["W"], pred_box["S"], pred_box["E"], pred_box["N"]],
                [ref_box["W"], ref_box["S"], ref_box["E"], ref_box["N"]],
            ))
        except (KeyError, TypeError):
            # Pred missing one of W/S/E/N or non-numeric → invalid prediction → 0.
            return 0.0

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
        result = id_r(pred_seq, ref_seq)
        score = result["identity_ratio"]
        if isinstance(score, str):  # "Zero length alignment" / "Zero length sequences" → invalid pred
            return 0.0
        return float(score)

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
        rouge_norm = rouge_l(pred_text, answer)["rougeLsum"] / 100.0
        bert_f1 = bert_score_fn(pred_text, answer)["bert_f1"]
        return float(freeform_geometric(rouge_norm, bert_f1))

    # ------- Auxiliary observability metrics (weight 0, all tasks) ---------

    async def _aux_rouge_lsum(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
        pred_text = self._extract_pred(completion, state)
        if not pred_text.strip() or not answer:
            return 0.0
        return float(rouge_l(pred_text, answer)["rougeLsum"])

    async def _aux_bert_f1(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
        pred_text = self._extract_pred(completion, state)
        if not pred_text.strip() or not answer:
            return 0.0
        return float(bert_score_fn(pred_text, answer)["bert_f1"])
