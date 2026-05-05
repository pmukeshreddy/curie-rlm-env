"""Stage 3 — CurieRubric integration tests.

Test plan (Stage 3b spec):
- Per-task tests (10 × 4 = 40)
- Hack-pattern tests (4)
- Auxiliary metric tests (3)
- Structural tests (8)
- Anti-hack absence tests (2)

USE REAL CURIE DATA only. Stub judge_client for LLMSim tasks (the data is real;
the judge is a test substitute for the external Gemini API).
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
import verifiers as vf
import yaml
from verifiers.envs.experimental.rlm_env import RLMEnv, RLMMonitorRubric

from curie_rlm_env import CurieRLMEnv, CurieRubric, load_environment
from curie_rlm_env.scorers import bert_score_fn, id_r, iou, llm_sim, rouge_l


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data" / "curie" / "data" / "data"
_CONFIG = _PROJECT_ROOT / "config"
_SRC_DIR = _PROJECT_ROOT / "src" / "curie_rlm_env"
_FROZEN_PROMPTS = _CONFIG / "frozen_prompts" / "llmsim"
_CURIE_PROMPTS = _PROJECT_ROOT / "data" / "curie" / "prompts"


TASK_FOLDER = {
    "DFT-S": "dft", "DFT-P": "dft", "DFT-C": "dft",
    "MPVE": "mpve", "BIOGR": "biogr", "PDB": "pdb",
    "HFE": "hfe", "HFD": "hfd", "QECC_65": "qecc_65", "GEO": "geo",
}


def _first_gt(folder: str) -> Path:
    gt_dir = _DATA_ROOT / folder / "ground_truth"
    paths = sorted(gt_dir.glob("*.json"))
    if not paths:
        pytest.fail(f"No ground_truth files in {gt_dir}")
    return paths[0]


def _gt_text(folder: str) -> str:
    return _first_gt(folder).read_text()


def _gt_obj(folder: str):
    return json.loads(_gt_text(folder))


def stub_judge_match(prompt: str) -> str:
    return '{"json_extracted_index": 0, "compare": {}}'


def stub_judge_no_match(prompt: str) -> str:
    return '{}'


@pytest.fixture(scope="module")
def rubric_match():
    return CurieRubric(judge_client=stub_judge_match)


@pytest.fixture(scope="module")
def rubric_no_match():
    return CurieRubric(judge_client=stub_judge_no_match)


@pytest.fixture(scope="module")
def rubric_default():
    return CurieRubric()


@pytest.fixture(scope="module")
def env_dft_s():
    return load_environment("DFT-S", "test")


def _make_state(task_id: str, completion: str, answer: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": "test"}],
        "completion": completion,
        "answer": answer,
        "task": task_id,
        "info": {"task_id": task_id},
        "final_answer": completion,
        "trajectory": [],
    }


def _aggregate(rubric: CurieRubric, state: dict) -> float:
    asyncio.run(rubric.score_group([state]))
    return state["reward"]


def _pdb_fasta(seq: str) -> str:
    return f">predicted_seq\n{seq}"


# ===========================================================================
# Guard #1 (per-task) — DFT-S
# ===========================================================================

def test_dft_s_perfect_match(rubric_match):
    ans = json.dumps([{"foo": "bar"}])
    s = _make_state("DFT-S", ans, ans)
    assert _aggregate(rubric_match, s) == pytest.approx(0.7, abs=0.05)


def test_dft_s_empty_input_zero(rubric_match):
    ans = json.dumps([{"foo": "bar"}])
    s = _make_state("DFT-S", "", ans)
    assert _aggregate(rubric_match, s) == 0.0


def test_dft_s_real_gt_example(rubric_match):
    gt = _gt_text(TASK_FOLDER["DFT-S"])
    s = _make_state("DFT-S", gt, gt)
    assert 0.65 <= _aggregate(rubric_match, s) <= 0.75


def test_dft_s_garbage_input_low(rubric_no_match):
    ans = json.dumps([{"foo": "bar"}])
    s = _make_state("DFT-S", json.dumps([{"name": "lorem"}]), ans)
    assert _aggregate(rubric_no_match, s) < 0.3


# ===========================================================================
# Guard #1 (per-task) — DFT-P
# ===========================================================================

def test_dft_p_perfect_match(rubric_match):
    ans = json.dumps([{"foo": "bar"}])
    s = _make_state("DFT-P", ans, ans)
    assert _aggregate(rubric_match, s) == pytest.approx(0.7, abs=0.05)


def test_dft_p_empty_input_zero(rubric_match):
    ans = json.dumps([{"foo": "bar"}])
    s = _make_state("DFT-P", "", ans)
    assert _aggregate(rubric_match, s) == 0.0


def test_dft_p_real_gt_example(rubric_match):
    gt = _gt_text(TASK_FOLDER["DFT-P"])
    s = _make_state("DFT-P", gt, gt)
    assert 0.65 <= _aggregate(rubric_match, s) <= 0.75


def test_dft_p_garbage_input_low(rubric_no_match):
    ans = json.dumps([{"foo": "bar"}])
    s = _make_state("DFT-P", json.dumps([{"name": "lorem"}]), ans)
    assert _aggregate(rubric_no_match, s) < 0.3


# ===========================================================================
# Guard #1 (per-task) — DFT-C (free-form)
# ===========================================================================

def test_dft_c_perfect_match(rubric_default):
    ans = "Some scientific answer text describing molecular dynamics."
    s = _make_state("DFT-C", ans, ans)
    assert _aggregate(rubric_default, s) >= 0.95


def test_dft_c_empty_input_zero(rubric_default):
    ans = "Some answer text"
    s = _make_state("DFT-C", "", ans)
    assert _aggregate(rubric_default, s) == 0.0


def test_dft_c_real_gt_example(rubric_default):
    gt = _gt_text(TASK_FOLDER["DFT-C"])
    s = _make_state("DFT-C", gt, gt)
    assert 0.95 <= _aggregate(rubric_default, s) <= 1.01


def test_dft_c_garbage_input_low(rubric_default):
    gt = _gt_text(TASK_FOLDER["DFT-C"])
    s = _make_state("DFT-C", "lorem ipsum xyz", gt)
    assert _aggregate(rubric_default, s) < 0.5


# ===========================================================================
# Guard #1 (per-task) — MPVE
# ===========================================================================

def test_mpve_perfect_match(rubric_match):
    ans = json.dumps([{"foo": "bar"}])
    s = _make_state("MPVE", ans, ans)
    assert _aggregate(rubric_match, s) == pytest.approx(0.7, abs=0.05)


def test_mpve_empty_input_zero(rubric_match):
    ans = json.dumps([{"foo": "bar"}])
    s = _make_state("MPVE", "", ans)
    assert _aggregate(rubric_match, s) == 0.0


def test_mpve_real_gt_example(rubric_match):
    gt = _gt_text(TASK_FOLDER["MPVE"])
    s = _make_state("MPVE", gt, gt)
    assert 0.65 <= _aggregate(rubric_match, s) <= 0.75


def test_mpve_garbage_input_low(rubric_no_match):
    ans = json.dumps([{"foo": "bar"}])
    s = _make_state("MPVE", json.dumps([{"name": "lorem"}]), ans)
    assert _aggregate(rubric_no_match, s) < 0.3


# ===========================================================================
# Guard #1 (per-task) — BIOGR (IoU)
# ===========================================================================

def test_biogr_perfect_match(rubric_default):
    bbox = json.dumps({"W": -10.0, "S": -10.0, "E": 10.0, "N": 10.0})
    s = _make_state("BIOGR", bbox, bbox)
    assert _aggregate(rubric_default, s) == pytest.approx(1.0, abs=0.01)


def test_biogr_empty_input_zero(rubric_default):
    bbox = json.dumps({"W": -10.0, "S": -10.0, "E": 10.0, "N": 10.0})
    s = _make_state("BIOGR", "", bbox)
    assert _aggregate(rubric_default, s) == 0.0


def test_biogr_real_gt_example(rubric_default):
    gt = _gt_text(TASK_FOLDER["BIOGR"])
    s = _make_state("BIOGR", gt, gt)
    assert 0.99 <= _aggregate(rubric_default, s) <= 1.0


def test_biogr_garbage_input_low(rubric_default):
    gt = _gt_text(TASK_FOLDER["BIOGR"])
    s = _make_state("BIOGR", "lorem ipsum xyz", gt)
    assert _aggregate(rubric_default, s) < 0.3


# ===========================================================================
# Guard #1 (per-task) — PDB (ID_r)
# ===========================================================================

def test_pdb_perfect_match(rubric_default):
    gt = _gt_text(TASK_FOLDER["PDB"])
    pred = _pdb_fasta(json.loads(gt)["sequence"])
    s = _make_state("PDB", pred, gt)
    assert _aggregate(rubric_default, s) >= 0.95


def test_pdb_empty_input_zero(rubric_default):
    gt = _gt_text(TASK_FOLDER["PDB"])
    s = _make_state("PDB", "", gt)
    assert _aggregate(rubric_default, s) == 0.0


def test_pdb_real_gt_example(rubric_default):
    gt = _gt_text(TASK_FOLDER["PDB"])
    pred = _pdb_fasta(json.loads(gt)["sequence"])
    s = _make_state("PDB", pred, gt)
    assert 0.95 <= _aggregate(rubric_default, s) <= 1.0


def test_pdb_garbage_input_low(rubric_default):
    gt = _gt_text(TASK_FOLDER["PDB"])
    s = _make_state("PDB", "lorem ipsum xyz", gt)
    assert _aggregate(rubric_default, s) < 0.3


# ===========================================================================
# Guard #1 (per-task) — HFE (free-form)
# ===========================================================================

def test_hfe_perfect_match(rubric_default):
    ans = "Hartree-Fock extraction yields the Hamiltonian for the lattice model."
    s = _make_state("HFE", ans, ans)
    assert _aggregate(rubric_default, s) >= 0.95


def test_hfe_empty_input_zero(rubric_default):
    ans = "some answer"
    s = _make_state("HFE", "", ans)
    assert _aggregate(rubric_default, s) == 0.0


def test_hfe_real_gt_example(rubric_default):
    gt = _gt_text(TASK_FOLDER["HFE"])
    s = _make_state("HFE", gt, gt)
    assert 0.95 <= _aggregate(rubric_default, s) <= 1.01


def test_hfe_garbage_input_low(rubric_default):
    gt = _gt_text(TASK_FOLDER["HFE"])
    s = _make_state("HFE", "lorem ipsum xyz", gt)
    assert _aggregate(rubric_default, s) < 0.5


# ===========================================================================
# Guard #1 (per-task) — HFD (free-form)
# ===========================================================================

def test_hfd_perfect_match(rubric_default):
    ans = "Derivation of the effective Hamiltonian via canonical transformation."
    s = _make_state("HFD", ans, ans)
    assert _aggregate(rubric_default, s) >= 0.95


def test_hfd_empty_input_zero(rubric_default):
    ans = "some answer"
    s = _make_state("HFD", "", ans)
    assert _aggregate(rubric_default, s) == 0.0


def test_hfd_real_gt_example(rubric_default):
    gt = _gt_text(TASK_FOLDER["HFD"])
    s = _make_state("HFD", gt, gt)
    assert 0.95 <= _aggregate(rubric_default, s) <= 1.01


def test_hfd_garbage_input_low(rubric_default):
    gt = _gt_text(TASK_FOLDER["HFD"])
    s = _make_state("HFD", "lorem ipsum xyz", gt)
    assert _aggregate(rubric_default, s) < 0.5


# ===========================================================================
# Guard #1 (per-task) — QECC_65 (free-form)
# ===========================================================================

def test_qecc_65_perfect_match(rubric_default):
    ans = "The stabilizer code uses 65 physical qubits to encode logical information."
    s = _make_state("QECC_65", ans, ans)
    assert _aggregate(rubric_default, s) >= 0.95


def test_qecc_65_empty_input_zero(rubric_default):
    ans = "some answer"
    s = _make_state("QECC_65", "", ans)
    assert _aggregate(rubric_default, s) == 0.0


def test_qecc_65_real_gt_example(rubric_default):
    gt = _gt_text(TASK_FOLDER["QECC_65"])
    s = _make_state("QECC_65", gt, gt)
    assert 0.95 <= _aggregate(rubric_default, s) <= 1.01


def test_qecc_65_garbage_input_low(rubric_default):
    gt = _gt_text(TASK_FOLDER["QECC_65"])
    s = _make_state("QECC_65", "lorem ipsum xyz", gt)
    assert _aggregate(rubric_default, s) < 0.5


# ===========================================================================
# Guard #1 (per-task) — GEO (free-form)
# ===========================================================================

def test_geo_perfect_match(rubric_default):
    ans = "The geographic dataset spans North America from 1990 to 2020."
    s = _make_state("GEO", ans, ans)
    assert _aggregate(rubric_default, s) >= 0.95


def test_geo_empty_input_zero(rubric_default):
    ans = "some answer"
    s = _make_state("GEO", "", ans)
    assert _aggregate(rubric_default, s) == 0.0


def test_geo_real_gt_example(rubric_default):
    gt = _gt_text(TASK_FOLDER["GEO"])
    s = _make_state("GEO", gt, gt)
    assert 0.95 <= _aggregate(rubric_default, s) <= 1.01


def test_geo_garbage_input_low(rubric_default):
    gt = _gt_text(TASK_FOLDER["GEO"])
    s = _make_state("GEO", "lorem ipsum xyz", gt)
    assert _aggregate(rubric_default, s) < 0.5


# ===========================================================================
# Hack-pattern tests
# ===========================================================================

def test_biogr_huge_box(rubric_default):
    # Predicted huge box — IoU's union denominator handles big-box hack naturally.
    gt = _gt_text(TASK_FOLDER["BIOGR"])
    huge = json.dumps({"W": -10000.0, "S": -10000.0, "E": 10000.0, "N": 10000.0})
    s = _make_state("BIOGR", huge, gt)
    assert _aggregate(rubric_default, s) < 0.05


def test_pdb_random_string(rubric_default):
    # Random 100-char AA string — ID_r should be low against real GT sequence.
    gt = _gt_text(TASK_FOLDER["PDB"])
    random_seq = "ACDEFGHIKLMNPQRSTVWY" * 5  # 100 chars, deterministic non-real seq
    s = _make_state("PDB", _pdb_fasta(random_seq), gt)
    assert _aggregate(rubric_default, s) < 0.5


def test_freeform_repetition(rubric_default):
    # 'the the the ...' × 50 — both ROUGE and BERT should flag low semantic content.
    gt = _gt_text(TASK_FOLDER["HFE"])
    s = _make_state("HFE", "the the the " * 50, gt)
    assert _aggregate(rubric_default, s) < 0.4


def test_freeform_short_confident(rubric_default):
    # Confident-sounding but content-empty — BERT should catch semantic emptiness.
    gt = _gt_text(TASK_FOLDER["HFE"])
    s = _make_state("HFE", "I am highly confident the answer is correct", gt)
    assert _aggregate(rubric_default, s) < 0.5


# ===========================================================================
# Auxiliary metric tests (weight=0, applied to ALL 10 tasks for observability)
# ===========================================================================

def test_aux_rouge_applied_to_all_10_tasks(rubric_default):
    func_names = [f.__name__ for f in rubric_default.funcs]
    assert "_aux_rouge_lsum" in func_names


def test_aux_bert_applied_to_all_10_tasks(rubric_default):
    func_names = [f.__name__ for f in rubric_default.funcs]
    assert "_aux_bert_f1" in func_names


def test_aux_metrics_weight_zero(rubric_default):
    aux_names = {"_aux_rouge_lsum", "_aux_bert_f1"}
    for func, weight in zip(rubric_default.funcs, rubric_default.weights):
        if func.__name__ in aux_names:
            assert weight == 0.0


# ===========================================================================
# Structural tests
# ===========================================================================

def test_curie_rubric_replaces_placeholder(env_dft_s):
    # CurieRLMEnv must wire CurieRubric (Stage 3b replaces vf.Rubric() placeholder)
    rubrics = (
        env_dft_s.rubric.rubrics
        if isinstance(env_dft_s.rubric, vf.RubricGroup)
        else [env_dft_s.rubric]
    )
    assert any(isinstance(r, CurieRubric) for r in rubrics)


def test_monitor_rubric_still_attached(env_dft_s):
    # RLMMonitorRubric must remain in chain after CurieRubric wiring (rlm_env.py:2517)
    rubrics = (
        env_dft_s.rubric.rubrics
        if isinstance(env_dft_s.rubric, vf.RubricGroup)
        else [env_dft_s.rubric]
    )
    assert any(isinstance(r, RLMMonitorRubric) for r in rubrics)


def test_all_10_tasks_routable(rubric_default):
    # Every task_id must be handled by exactly one headline reward family
    expected = {"DFT-S", "DFT-P", "DFT-C", "MPVE", "BIOGR", "PDB",
                "HFE", "HFD", "QECC_65", "GEO"}
    from curie_rlm_env.rubric import (
        _RETRIEVAL_TASKS, _FREEFORM_TASKS, _GEOMETRIC_TASKS, _STRUCTURAL_TASKS,
    )
    union = _RETRIEVAL_TASKS | _FREEFORM_TASKS | _GEOMETRIC_TASKS | _STRUCTURAL_TASKS
    assert union == expected


def test_dft_c_routes_to_freeform():
    # Stage 3b decision: DFT-C is free-form, NOT retrieval (per Curie release)
    from curie_rlm_env.rubric import _RETRIEVAL_TASKS, _FREEFORM_TASKS
    assert "DFT-C" in _FREEFORM_TASKS
    assert "DFT-C" not in _RETRIEVAL_TASKS


def test_reward_weights_match_yaml(rubric_default):
    cfg = yaml.safe_load((_CONFIG / "safeguards.yaml").read_text())["rubric"]
    weights = {f.__name__: w for f, w in zip(rubric_default.funcs, rubric_default.weights)}
    assert weights["_llmsim_reward"] == cfg["llm_sim_weight"]
    assert weights["_iou_reward"] == cfg["deterministic_weight"]
    assert weights["_idr_reward"] == cfg["deterministic_weight"]
    # ROUGE/BERT free-form weights are 0.5 each (sum 1.0); 0.5 == freeform_weight (renamed from lm_score_weight in Stage 3 cleanup)
    assert weights["_rouge_freeform_reward"] == cfg["freeform_weight"]
    assert weights["_bert_freeform_reward"] == cfg["freeform_weight"]


def test_judge_model_is_gemini_2_5_pro():
    cfg = yaml.safe_load((_CONFIG / "judge.yaml").read_text())
    assert cfg["judge_model_id"] == "gemini-2.5-pro"


def test_frozen_prompts_byte_exact():
    # Stage 3b: frozen prompts must match Curie's verbatim
    pairs = [
        ("dft_structure.txt", "dft_structure_eval_output_1_shot.txt"),
        ("dft_metadata.txt", "dft_metadata_eval_output_1_shot.txt"),
        ("mat.txt", "mat_eval_output_1_shot.txt"),
    ]
    for ours, theirs in pairs:
        ours_bytes = (_FROZEN_PROMPTS / ours).read_bytes()
        theirs_bytes = (_CURIE_PROMPTS / theirs).read_bytes()
        assert ours_bytes == theirs_bytes, f"{ours} differs from {theirs}"


def test_pdb_code_exec_branch_absent():
    # Stage 3b: code-exec branch dropped for sandbox safety
    for f in ("scorers.py", "rubric.py", "env.py"):
        content = (_SRC_DIR / f).read_text()
        # Strip comments before checking — comments referencing exec are OK
        code_lines = [
            line for line in content.splitlines()
            if not line.strip().startswith("#")
        ]
        code = "\n".join(code_lines)
        assert "exec(" not in code, f"exec( call found in {f} (non-comment)"


# ===========================================================================
# Anti-hack absence tests (locks the design decision)
# ===========================================================================

def test_no_length_penalty_function():
    # Stage 3b: zero anti-hack reward functions — no length_penalty / substring_guard /
    # big_box_guard / recursion_required / consistency_check anywhere in src/.
    forbidden = ["length_penalty", "substring_guard", "big_box_guard",
                 "recursion_required", "consistency_check"]
    for py_file in _SRC_DIR.glob("*.py"):
        content = py_file.read_text()
        # Strip comments
        code_lines = [line for line in content.splitlines() if not line.strip().startswith("#")]
        code = "\n".join(code_lines)
        for word in forbidden:
            assert word not in code, f"Forbidden anti-hack name '{word}' in {py_file.name}"


def test_zero_anti_hack_reward_functions(rubric_default):
    # CurieRubric must contain ONLY the 5 headline reward funcs + 2 aux metrics. No others.
    func_names = sorted(f.__name__ for f in rubric_default.funcs)
    expected = sorted([
        "_llmsim_reward", "_iou_reward", "_idr_reward",
        "_rouge_freeform_reward", "_bert_freeform_reward",
        "_aux_rouge_lsum", "_aux_bert_f1",
    ])
    assert func_names == expected


# ===========================================================================
# Strict failure semantics: invalid PREDICTION → 0; invalid REFERENCE → raise.
# ===========================================================================


def _call_reward(rubric: CurieRubric, fn_name: str, task_id: str, completion: str, answer: str):
    fn = getattr(rubric, fn_name)
    state = _make_state(task_id, completion, answer)
    return asyncio.run(fn(
        prompt=state["prompt"], completion=completion, answer=answer,
        state=state, task=task_id, info=state["info"],
    ))


def test_llmsim_reward_raises_on_malformed_reference(rubric_match):
    with pytest.raises(ValueError):
        _call_reward(rubric_match, "_llmsim_reward", "DFT-S",
                     completion='[{"foo": "bar"}]', answer="this is not JSON {{")


def test_llmsim_reward_raises_on_empty_reference(rubric_match):
    with pytest.raises(ValueError):
        _call_reward(rubric_match, "_llmsim_reward", "DFT-S",
                     completion='[{"foo": "bar"}]', answer="")


def test_llmsim_reward_returns_zero_on_malformed_prediction(rubric_match):
    """Strict: invalid PREDICTION still returns 0 silently — model's fault, not data's."""
    score = _call_reward(rubric_match, "_llmsim_reward", "DFT-S",
                         completion="not parseable JSON {{", answer='[{"foo": "bar"}]')
    assert score == 0.0


def test_iou_reward_raises_on_non_dict_reference(rubric_default):
    with pytest.raises(ValueError):
        _call_reward(rubric_default, "_iou_reward", "BIOGR",
                     completion='{"W":0,"S":0,"E":1,"N":1}', answer='[1,2,3,4]')


def test_iou_reward_raises_on_missing_reference_keys(rubric_default):
    with pytest.raises(ValueError):
        _call_reward(rubric_default, "_iou_reward", "BIOGR",
                     completion='{"W":0,"S":0,"E":1,"N":1}', answer='{"W":0,"S":0,"E":1}')


def test_iou_reward_returns_zero_on_malformed_prediction(rubric_default):
    score = _call_reward(rubric_default, "_iou_reward", "BIOGR",
                         completion="not json", answer='{"W":0,"S":0,"E":1,"N":1}')
    assert score == 0.0


def test_idr_reward_raises_on_missing_sequence_field(rubric_default):
    with pytest.raises(ValueError):
        _call_reward(rubric_default, "_idr_reward", "PDB",
                     completion=_pdb_fasta("ACGT"), answer='{"other": "field"}')


def test_idr_reward_raises_on_empty_sequence(rubric_default):
    with pytest.raises(ValueError):
        _call_reward(rubric_default, "_idr_reward", "PDB",
                     completion=_pdb_fasta("ACGT"), answer='{"sequence": ""}')


def test_freeform_rouge_raises_on_empty_reference(rubric_default):
    with pytest.raises(ValueError):
        _call_reward(rubric_default, "_rouge_freeform_reward", "HFE",
                     completion="some prediction text", answer="")


def test_freeform_bert_raises_on_empty_reference(rubric_default):
    with pytest.raises(ValueError):
        _call_reward(rubric_default, "_bert_freeform_reward", "HFE",
                     completion="some prediction text", answer="")


def test_freeform_returns_zero_on_empty_prediction(rubric_default):
    """Strict: empty prediction with valid reference → 0 silently."""
    score = _call_reward(rubric_default, "_rouge_freeform_reward", "HFE",
                         completion="", answer="real reference text here")
    assert score == 0.0


# ===========================================================================
# Strict LLMSim: malformed judge JSON raises (no repair fallback).
# ===========================================================================


def test_llm_sim_raises_on_malformed_judge_output():
    """scorers.llm_sim must raise when judge_client returns invalid JSON; no repair."""
    from curie_rlm_env.judge_cache import clear_cache

    prompt_path = _FROZEN_PROMPTS / "dft_structure.txt"
    if not prompt_path.is_file():
        pytest.skip("LLMSim prompt file not present in this checkout")

    # Clear judge cache to avoid being short-circuited by an earlier test that
    # cached a valid judge response for the same (gt, pred) key.
    clear_cache()

    def malformed_judge(_prompt: str) -> str:
        return "this is not JSON, {{"

    with pytest.raises(ValueError) as exc_info:
        llm_sim(
            json_pred=[{"strict_failure_audit_unique_pred_key": "v1"}],
            json_ref=[{"strict_failure_audit_unique_ref_key": "v1"}],
            prompt_path=str(prompt_path),
            judge_client=malformed_judge,
        )
    assert "malformed JSON" in str(exc_info.value)
    clear_cache()


def test_scorers_llm_sim_does_not_have_repair_branch():
    """Defensive: source must not contain the deleted JSON-repair regex pattern."""
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "curie_rlm_env" / "scorers.py"
    ).read_text()
    assert 're.finditer(r",\\s*\\{"' not in src, "scorers.py still has the JSON-repair regex"
    assert "= json5.loads(output[: inds[-1]] + \"]\")" not in src
