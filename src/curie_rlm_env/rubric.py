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


# Free-form GT field-extraction helpers. The dataset's `answer` for a free-form
# task is `json.dumps(entry["ground_truth"])` (datasets.py:_row_from_split_entry),
# and the ground_truth is a structured dict/list — `code` for DFT-C, Hamiltonian
# + Other_info for HFE, list of step dicts for HFD, list of catalog dicts for
# QECC_65, paper metadata + notes for GEO. Comparing prose summaries against
# `json.dumps(...)` of those structures passes JSON syntax (`{`, `}`, `"key":`)
# and identifier metadata (record_id, arxiv_id, paper_link) into ROUGE/BERT,
# both of which are noise. Extract content-only text instead.
_IDENTIFIER_KEY_NAMES = frozenset({"id", "url", "doi", "paper_link"})
_IDENTIFIER_KEY_SUFFIXES = ("_id",)


def _is_identifier_key(key) -> bool:
    """True if a GT key looks like metadata (record_id, arxiv_id, paper_link,
    etc.) rather than content. Used by _freeform_reference."""
    if not isinstance(key, str):
        return False
    k = key.lower()
    return k in _IDENTIFIER_KEY_NAMES or any(k.endswith(s) for s in _IDENTIFIER_KEY_SUFFIXES)


def _collect_content_strings(obj) -> list[str]:
    """Recursively collect string-typed values from a parsed GT (dict/list),
    skipping identifier-like keys. Order is the GT's natural traversal order
    so the joined reference text reads coherently."""
    if isinstance(obj, str):
        return [obj] if obj.strip() else []
    if isinstance(obj, dict):
        out: list[str] = []
        for k, v in obj.items():
            if _is_identifier_key(k):
                continue
            out.extend(_collect_content_strings(v))
        return out
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(_collect_content_strings(item))
        return out
    return []


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
    def _freeform_reference(answer_string: str, task_id: str, info=None) -> str:
        """Extract content-only reference text from a json-encoded free-form GT.

        Per-task field whitelists, NOT a generic "concatenate every string
        value" collector — that approach treats `Hamiltonian` (LaTeX) the same
        as `Score` (numeric string) for HFE and concatenates equation strings
        with step prose for HFD, both of which inject scoring noise. Each task
        names exactly the fields whose content the model is meant to match.

        Field choices come from the GT shapes observed in
        scripts/investigate_gt_format.py and the file structure of
        data/curie/data/data/{folder}/ground_truth/. Where the choice is not
        unambiguously implied by TASK_MAP or the field semantics, the comment
        flags it as an ASSUMPTION so a future audit against
        data/curie/colabs/curie_run_eval.ipynb cell 30 (the upstream Curie
        scorer that defines `_FULL_ADDITIONAL_METRICS`) can confirm or
        correct it.

        Falls back to the original `answer_string` when the GT can't be parsed
        as JSON or the per-task extraction yields no content. Preserving the
        pre-fix behavior on an unexpected shape is preferable to returning an
        empty reference (which would silently zero the reward).
        """
        try:
            gt = json5.loads(answer_string)
        except (ValueError, TypeError):
            return answer_string

        parts: list[str] = []

        # DFT-C: GT is a dict like {code, graph_as_text, no_header_code,
        # record_id}. The canonical answer field name is carried on the
        # dataset row as info["dft_field"] (== "code" for DFT-C, set by
        # datasets.TASK_MAP). Reading info instead of hardcoding "code" keeps
        # the rubric correct if a future DFT-Cx variant ships with a
        # different field name. NOT an ASSUMPTION — TASK_MAP is the source
        # of truth here.
        if task_id == "DFT-C":
            field = info["dft_field"] if isinstance(info, dict) and info.get("dft_field") else "code"
            if isinstance(gt, dict):
                v = gt.get(field)
                if isinstance(v, str) and v.strip():
                    parts.append(v)

        # HFE: GT is a dict like {Hamiltonian, Other_info, Score, arxiv_id,
        # record_id}. The model's task is to reproduce the Hamiltonian (LaTeX
        # equations) plus the prose context. Score is a single numeric
        # quality string from Curie's curation process (excluded — would
        # inject token-noise into ROUGE/BERT). arxiv_id and record_id are
        # identifier metadata (excluded). ASSUMPTION pending cell 30
        # verification: confirm that Curie's free-form scorer for HFE uses
        # both Hamiltonian and Other_info as the reference (vs. e.g.
        # Hamiltonian alone).
        elif task_id == "HFE":
            if isinstance(gt, dict):
                for key in ("Hamiltonian", "Other_info"):
                    v = gt.get(key)
                    if isinstance(v, str) and v.strip():
                        parts.append(v)

        # HFD: GT is a list of step dicts. Per-step prose lives in `task`
        # (description of what the step does) and `Note` (auxiliary prose).
        # Other per-step fields like `Step` (the integer step number as a
        # string) and `Equation` (LaTeX) are excluded — Step is positional
        # noise, Equation is non-prose that compresses BERT. ASSUMPTION
        # pending cell 30 verification: confirm Curie's HFD scorer uses
        # task + Note as the reference (vs. concatenating all step fields).
        elif task_id == "HFD":
            if isinstance(gt, list):
                for step in gt:
                    if isinstance(step, dict):
                        for key in ("task", "Note"):
                            v = step.get(key)
                            if isinstance(v, str) and v.strip():
                                parts.append(v)

        # QECC_65: GT is a list of catalog dicts like {code_id, physical,
        # logical, name, introduced, ...}. The descriptive fields are name,
        # physical, logical, introduced; code_id is the catalog identifier
        # (excluded). ASSUMPTION pending cell 30 verification: confirm
        # Curie's QECC_65 scorer uses these descriptive fields as the
        # reference (vs. e.g. only `name` or only `physical`).
        elif task_id == "QECC_65":
            if isinstance(gt, list):
                for record in gt:
                    if isinstance(record, dict):
                        for key in ("name", "physical", "logical", "introduced"):
                            v = record.get(key)
                            if isinstance(v, str) and v.strip():
                                parts.append(v)

        # GEO: GT is a dict like {notes, paper_title, paper_link, datasets,
        # record_id}. The prose answer is `notes`. paper_title is included
        # because BERT/ROUGE benefit from the topic anchor. paper_link and
        # record_id are identifier metadata (excluded). datasets is excluded
        # because it's typically a list of dataset identifiers, not prose.
        # ASSUMPTION pending cell 30 verification: confirm Curie's GEO
        # scorer uses notes (+/- paper_title) as the reference (vs. notes
        # alone or all string fields).
        elif task_id == "GEO":
            if isinstance(gt, dict):
                for key in ("notes", "paper_title"):
                    v = gt.get(key)
                    if isinstance(v, str) and v.strip():
                        parts.append(v)

        if parts:
            return "\n".join(parts)

        # Per-task extraction returned nothing — likely an unexpected GT
        # shape (e.g. an upstream schema change). Fall back to the generic
        # content collector before giving up entirely; if that also returns
        # nothing, surface the original answer_string. Both layers preserve
        # behavior on unknown shapes instead of zeroing the reward.
        generic = _collect_content_strings(gt)
        if generic:
            return "\n".join(generic)
        return answer_string

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
        # Per-task content extraction. The dataset hands us
        # `json.dumps(structured_gt)`; passing that directly to ROUGE/BERT mixes
        # JSON syntax (`{`, `}`, `"key":`) and identifier metadata (record_id,
        # arxiv_id, paper_link) into the comparison. _freeform_reference uses
        # an explicit per-task field whitelist (see method docstring) and
        # threads `info` through so DFT-C reads info["dft_field"] from the
        # dataset row instead of hardcoding the field name.
        ref_text = self._freeform_reference(answer, task_id, info)
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
            self._freeform_reference(answer, task_id, info)
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
            self._freeform_reference(answer, task_id, info)
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
