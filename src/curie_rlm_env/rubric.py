"""Stage 3b — CurieRubric per-task scoring dispatcher.

Reward weights per CLAUDE.md L21-25 + config/rubric_dispatcher.yaml + Stage 3b
locked decisions:
- DFT-S, DFT-P, MPVE: LLMSim × 0.7  (max 0.7)
- BIOGR: IoU × 1.0
- PDB: ID_r × 1.0  (FASTA `>` path only; code-exec branch dropped)
- DFT-C, HFE, HFD, QECC_65, GEO: ROUGE-Lsum/100 × 0.5 + BERT-F1 × 0.5  (max 1.0)

Auxiliary metrics (weight 0, applied to ALL 10 tasks for observability via
RubricGroup aggregate): rouge_lsum, bert_f1.

NO anti-hack reward functions per Stage 3b: Curie's formulas are length-bounded
structurally; observability via vf.MonitorRubric (auto-attached by RLMEnv) and
the weight-0 aux metrics provides hack-detection signal without preempting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import json5
import verifiers as vf

from .scorers import bert_score_fn, id_r, iou, llm_sim, rouge_l


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
        self.add_reward_func(self._rouge_freeform_reward, weight=0.5)
        self.add_reward_func(self._bert_freeform_reward, weight=0.5)
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
    def _safe_loads(text: str):
        """Parse JSON-ish text. Returns None on failure (input-validation, not fallback)."""
        if not text:
            return None
        try:
            return json5.loads(text)
        except (ValueError, TypeError):
            return None

    # ------- Headline reward funcs -----------------------------------------

    async def _llmsim_reward(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
        task_id = self._task_id(info, task)
        if task_id not in _RETRIEVAL_TASKS:
            return 0.0
        if self._judge_client is None:
            raise ValueError(f"LLMSim for {task_id} requires judge_client; got None")
        pred_text = self._extract_pred(completion, state)
        if not pred_text or not pred_text.strip():
            return 0.0
        json_pred = self._safe_loads(pred_text)
        if json_pred is None:
            return 0.0
        json_ref = self._safe_loads(answer)
        if json_ref is None:
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
        pred_text = self._extract_pred(completion, state)
        if not pred_text.strip():
            return 0.0
        pred_box = self._safe_loads(pred_text)
        ref_box = self._safe_loads(answer)
        if not isinstance(pred_box, dict) or not isinstance(ref_box, dict):
            return 0.0
        try:
            return float(iou(
                [pred_box["W"], pred_box["S"], pred_box["E"], pred_box["N"]],
                [ref_box["W"], ref_box["S"], ref_box["E"], ref_box["N"]],
            ))
        except (KeyError, TypeError):
            return 0.0

    async def _idr_reward(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
        task_id = self._task_id(info, task)
        if task_id not in _STRUCTURAL_TASKS:
            return 0.0
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
        ref_obj = self._safe_loads(answer)
        if not isinstance(ref_obj, dict):
            return 0.0
        ref_seq = ref_obj.get("sequence", "")
        if not isinstance(ref_seq, str) or not ref_seq:
            return 0.0
        result = id_r(pred_seq, ref_seq)
        score = result["identity_ratio"]
        if isinstance(score, str):  # "Zero length alignment" / "Zero length sequences"
            return 0.0
        return float(score)

    async def _rouge_freeform_reward(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
        task_id = self._task_id(info, task)
        if task_id not in _FREEFORM_TASKS:
            return 0.0
        pred_text = self._extract_pred(completion, state)
        if not pred_text.strip() or not answer:
            return 0.0
        return float(rouge_l(pred_text, answer)["rougeLsum"] / 100.0)

    async def _bert_freeform_reward(self, prompt, completion, answer, state, task, info, **kwargs) -> float:
        task_id = self._task_id(info, task)
        if task_id not in _FREEFORM_TASKS:
            return 0.0
        pred_text = self._extract_pred(completion, state)
        if not pred_text.strip() or not answer:
            return 0.0
        return float(bert_score_fn(pred_text, answer)["bert_f1"])

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
