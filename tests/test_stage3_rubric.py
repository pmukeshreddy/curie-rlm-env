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
from curie_rlm_env.scorers import (
    FREEFORM_ROUGE_EXPONENT,
    bert_score_fn,
    freeform_geometric,
    id_r,
    iou,
    llm_sim,
    rouge_l,
)


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
# Guard #7 — Free-form geometric coupling (anti-length-grift)
# Direct unit tests on scorers.freeform_geometric (no rubric, no fixtures).
# ===========================================================================

def test_freeform_geometric_alpha_is_zero_six():
    # Lock the exponent — the asymmetric weight toward ROUGE-L is the whole point.
    assert FREEFORM_ROUGE_EXPONENT == 0.6


def test_freeform_geometric_perfect():
    assert freeform_geometric(1.0, 1.0) == 1.0


def test_freeform_geometric_zero_on_rouge_zero():
    # Zero on either component must propagate; no epsilon, no floor.
    assert freeform_geometric(0.0, 1.0) == 0.0


def test_freeform_geometric_zero_on_bert_zero():
    assert freeform_geometric(1.0, 0.0) == 0.0


def test_freeform_geometric_both_half():
    # 0.5^0.6 * 0.5^0.4 = 0.5^1.0 = 0.5 exactly.
    assert freeform_geometric(0.5, 0.5) == pytest.approx(0.5, abs=1e-12)


def test_freeform_geometric_asymmetric_length_grift_regression():
    """Regression guard: geometric coupling penalizes length-grift harder than 0.5+0.5 additive.

    (r=0.1, b=0.8) is the canonical length-grift pattern: low lexical overlap
    (ROUGE), high semantic similarity (BERT) from padded filler. Old additive
    rewarded this at 0.45; new geometric returns ~0.230 — the incentive is gone.
    """
    r, b = 0.1, 0.8
    additive_old = 0.5 * r + 0.5 * b  # 0.45
    geometric_new = freeform_geometric(r, b)
    assert geometric_new == pytest.approx(0.22974, abs=1e-4)
    assert geometric_new < additive_old


def test_freeform_geometric_raises_above_one():
    with pytest.raises(ValueError):
        freeform_geometric(1.5, 0.5)


def test_freeform_geometric_raises_below_zero():
    with pytest.raises(ValueError):
        freeform_geometric(-0.1, 0.5)


def test_freeform_geometric_bounded_by_weighted_arithmetic_mean():
    """Young's inequality / weighted AM-GM: r^0.6 * b^0.4 <= 0.6*r + 0.4*b ∀ (r,b) in [0,1]^2."""
    grid = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    for r in grid:
        for b in grid:
            geo = freeform_geometric(r, b)
            am = 0.6 * r + 0.4 * b
            assert geo <= am + 1e-9, (
                f"freeform_geometric({r}, {b}) = {geo} > weighted AM {am}"
            )


def test_bert_rescaling_is_engaged():
    """Sanity: rescaled BERTScore must differ from raw by > 0.1 on a garbage fixture.

    Proves rescale_with_baseline=True is actually wired through bert_score_fn.
    Fixture is the same 'lorem ipsum xyz' vs HFE-perfect_match sentence pair
    already used by test_freeform_short_confident / test_hfe_garbage_input_low.
    Empirically: raw F1 ~0.78, rescaled F1 ~-0.30 (returned raw, not clamped),
    delta ~1.08.
    """
    pred = "lorem ipsum xyz"
    ref = "Hartree-Fock extraction yields the Hamiltonian for the lattice model."
    rescaled = bert_score_fn(pred, ref)["bert_f1"]
    # Parallel un-cached BERTScorer for the comparison (different instance, so
    # the module-level _BERT_SCORER singleton isn't disturbed).
    from bert_score import BERTScorer
    raw_scorer = BERTScorer(lang="en", rescale_with_baseline=False, device="cpu")
    _, _, raw_F1 = raw_scorer.score([pred], [ref])
    raw = raw_F1.item()
    assert raw - rescaled > 0.1, (
        f"rescaling not engaged: raw={raw}, rescaled={rescaled}"
    )


def test_bert_score_fn_returns_negative_for_below_baseline():
    """Diagnostic preservation: bert_score_fn must return the raw rescaled F1
    (including the negative tail) so _aux_bert_f1 logs the full distribution.

    Without this, every below-baseline rollout logs as 0.0 in Stage 5 W&B and
    we lose the ability to distinguish 'crossing baseline by -0.05' from
    'crashing through to -0.45' — the exact length-grift detection signal.
    """
    pred = "lorem ipsum xyz"
    ref = "Hartree-Fock extraction yields the Hamiltonian for the lattice model."
    bert_f1 = bert_score_fn(pred, ref)["bert_f1"]
    assert bert_f1 < 0.0, (
        f"bert_score_fn must return raw rescaled F1 (negative for below-baseline); got {bert_f1}"
    )


def test_freeform_geometric_clamps_bert_negative_to_zero(rubric_default):
    """Clamp moved to consumption site (CurieRubric._freeform_geometric_reward):
    when bert_score_fn returns a below-baseline (negative) value, the rubric
    clamps it to 0 before passing to freeform_geometric, and the zero-guard
    collapses the headline reward to 0.

    Aux _aux_bert_f1 still sees the raw negative value (see
    test_bert_score_fn_returns_negative_for_below_baseline) — only the
    headline geometric path clamps.
    """
    pred = "lorem ipsum xyz"
    ref = "Hartree-Fock extraction yields the Hamiltonian for the lattice model."
    s = _make_state("HFE", pred, ref)
    assert _aggregate(rubric_default, s) == 0.0


def test_freeform_geometric_rouge_strict_no_slop():
    """ROUGE input is validated STRICTLY; only BERT gets IEEE-754 slop.

    rouge_score returns 0-100 exactly (pure-Python fmeasure), divided by 100
    upstream — strict [0, 1] by construction. Slop on ROUGE would absorb real
    upstream bugs (e.g., a forgotten /100). BERT keeps slop because its
    cosine-similarity arithmetic overshoots 1.0 by ~1.2e-7 on identity matches.
    """
    # ROUGE strict: 1.0 + 5e-7 (within the old global slop) must now raise.
    with pytest.raises(ValueError):
        freeform_geometric(1.0 + 5e-7, 0.5)
    # BERT slop: same overshoot on the BERT input is permitted.
    result = freeform_geometric(0.5, 1.0 + 5e-7)
    # And the value is snapped to [0,1] post-validation so output ≤ 1.0.
    assert result <= 1.0


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
    # rescaled BERT (anti-length-grift, CLAUDE.md L57-59): garbage clamps to 0.0,
    # geometric zero-guard collapses the reward; tightened from <0.5 to <0.1.
    assert _aggregate(rubric_default, s) < 0.1


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
    # rescaled BERT (anti-length-grift, CLAUDE.md L57-59): tightened <0.5 -> <0.1.
    assert _aggregate(rubric_default, s) < 0.1


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
    # rescaled BERT (anti-length-grift, CLAUDE.md L57-59): tightened <0.5 -> <0.1.
    assert _aggregate(rubric_default, s) < 0.1


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
    # rescaled BERT (anti-length-grift, CLAUDE.md L57-59): tightened <0.5 -> <0.1.
    assert _aggregate(rubric_default, s) < 0.1


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
    # rescaled BERT (anti-length-grift, CLAUDE.md L57-59): tightened <0.5 -> <0.1.
    assert _aggregate(rubric_default, s) < 0.1


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
    # rescaled BERT (anti-length-grift, CLAUDE.md L57-59): rescaled F1 ~-0.48 clamps to 0,
    # zero-guard collapses reward to 0. Tightened from <0.4 to <0.05.
    gt = _gt_text(TASK_FOLDER["HFE"])
    s = _make_state("HFE", "the the the " * 50, gt)
    assert _aggregate(rubric_default, s) < 0.05


def test_freeform_short_confident(rubric_default):
    # Confident-sounding but content-empty — BERT should catch semantic emptiness.
    # rescaled BERT (anti-length-grift, CLAUDE.md L57-59): rescaled F1 ~0.07 (positive,
    # doesn't clamp); geometric mean with low ROUGE remains small. Tightened <0.5 -> <0.2.
    gt = _gt_text(TASK_FOLDER["HFE"])
    s = _make_state("HFE", "I am highly confident the answer is correct", gt)
    assert _aggregate(rubric_default, s) < 0.2


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
    # updated for geometric coupling (anti-length-grift):
    # _freeform_geometric_reward wired at deterministic_weight=1.0 (parity with IoU/ID_r);
    # cfg["freeform_weight"] is LEGACY (no longer wired) — see safeguards.yaml comment.
    assert weights["_freeform_geometric_reward"] == cfg["deterministic_weight"]


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
    # updated for geometric coupling (anti-length-grift): ROUGE+BERT free-form pair
    # collapsed into a single _freeform_geometric_reward. Now 4 headline + 2 aux = 6.
    func_names = sorted(f.__name__ for f in rubric_default.funcs)
    expected = sorted([
        "_llmsim_reward", "_iou_reward", "_idr_reward",
        "_freeform_geometric_reward",
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


def test_freeform_geometric_raises_on_empty_reference(rubric_default):
    # updated for geometric coupling (anti-length-grift): old ROUGE/BERT pair
    # collapsed into one reward func; the empty-reference invariant is unchanged.
    with pytest.raises(ValueError):
        _call_reward(rubric_default, "_freeform_geometric_reward", "HFE",
                     completion="some prediction text", answer="")


def test_freeform_returns_zero_on_empty_prediction(rubric_default):
    """Strict: empty prediction with valid reference → 0 silently."""
    # updated for geometric coupling (anti-length-grift): function name only.
    score = _call_reward(rubric_default, "_freeform_geometric_reward", "HFE",
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


def test_llm_sim_raises_on_empty_judge_list():
    """scorers.llm_sim must raise when judge_client returns an empty JSON list.

    Empty list is outside the judge prompt's contract (which yields either a
    matched record or a non-list null signal). Silently coercing to `{}` would
    score it as "no match" and lose the diagnostic — a real prompt-confusion
    failure mode would be indistinguishable from a legitimate zero score.
    """
    from curie_rlm_env.judge_cache import clear_cache

    prompt_path = _FROZEN_PROMPTS / "dft_structure.txt"
    if not prompt_path.is_file():
        pytest.skip("LLMSim prompt file not present in this checkout")

    clear_cache()

    def empty_list_judge(_prompt: str) -> str:
        return "[]"

    with pytest.raises(ValueError) as exc_info:
        llm_sim(
            json_pred=[{"empty_list_audit_unique_pred_key": "v1"}],
            json_ref=[{"empty_list_audit_unique_ref_key": "v1"}],
            prompt_path=str(prompt_path),
            judge_client=empty_list_judge,
        )
    msg = str(exc_info.value)
    assert "empty JSON list" in msg
    assert "gt index 0" in msg
    clear_cache()


# ===========================================================================
# Guard #2 — LLMSim post-LLM numeric verifier (anti-verbosity-grift)
# ===========================================================================

def test_extract_numeric_values_per_unit_family():
    """One representative per in-scope unit family."""
    from curie_rlm_env.scorers import _extract_numeric_values
    assert _extract_numeric_values({"bandgap": "2.1 eV"}) == {"bandgap": (2.1, "eV")}
    assert _extract_numeric_values({"freq": "21000 cm⁻¹"}) == {"freq": (21000.0, "cm⁻¹")}
    assert _extract_numeric_values({"temp": "300 K"}) == {"temp": (300.0, "K")}
    assert _extract_numeric_values({"latt": "5.4 Å"}) == {"latt": (5.4, "Å")}


def test_extract_numeric_values_within_record_verbosity_grift():
    """Regression guard: same canonical value despite prose dressing.

    The verbosity-grift attack: pad a wrong numeric with plausible prose to
    fool Gemini into accepting it. Extractor must return the SAME (mag, unit)
    pair as the bare value — only then can the verifier compare apples-to-apples.
    """
    from curie_rlm_env.scorers import _extract_numeric_values
    bare = _extract_numeric_values({"bandgap": "2.1 eV"})
    dressed = _extract_numeric_values(
        {"bandgap": "2.1 eV, measured at 300K via photoluminescence, reported in Table 3"}
    )
    assert bare == dressed == {"bandgap": (2.1, "eV")}


def test_extract_numeric_values_unparseable_field_is_omitted():
    """Unparseable values are OMITTED — not coerced to 0, not None.

    Verifier downstream relies on "absent from extraction" meaning "no opinion
    on this field". Coercing to 0 would silently treat "see Figure 3" as a
    zero-magnitude match candidate.
    """
    from curie_rlm_env.scorers import _extract_numeric_values
    result = _extract_numeric_values({"v": "see Figure 3"})
    assert result == {}
    assert "v" not in result  # explicit: not absent-with-None, just absent


def test_verify_numeric_match_cases():
    """Each case from the spec, with hand-computed relative errors."""
    from curie_rlm_env.scorers import _verify_numeric_match
    # Identical → True
    assert _verify_numeric_match({"x": "2.0 eV"}, {"x": "2.0 eV"})
    # 4% disagreement → True (0.08/2.08 = 0.0385 < 0.05)
    assert _verify_numeric_match({"x": "2.0 eV"}, {"x": "2.08 eV"})
    # 6% disagreement → False (0.12/2.12 = 0.0566 > 0.05)
    assert not _verify_numeric_match({"x": "2.0 eV"}, {"x": "2.12 eV"})
    # Different units, different families → False
    assert not _verify_numeric_match({"x": "2.0 eV"}, {"x": "2.0 Å"})
    # Different unit, convertible alias (electron-volts → eV) → True
    assert _verify_numeric_match({"x": "2.1 eV"}, {"x": "2.1 electron-volts"})
    # Within-family multiplicative conversion (meV → eV): 2000 meV = 2.0 eV → True
    assert _verify_numeric_match({"x": "2000 meV"}, {"x": "2.0 eV"})
    # No overlapping numeric fields → True (verifier abstains)
    assert _verify_numeric_match({"name": "abc"}, {"bandgap": "2.1 eV"})


def _always_match_judge(_prompt: str) -> str:
    """Mock judge: unconditionally claims a match against pred index 0."""
    return '{"json_extracted_index": 0, "compare": {}}'


def test_llm_sim_verifier_revokes_false_match_on_numeric_disagreement():
    """Gemini says match, numerics disagree by 20% → verifier revokes.

    ref = [{"bandgap": "2.0 eV"}], pred = [{"bandgap": "2.5 eV"}]
    rel_error = 0.5 / 2.5 = 0.20 > 0.05 → revoke.
    Expected: num_match=0, verifier_revoked_count=1, f1=0.
    """
    from curie_rlm_env.judge_cache import clear_cache

    prompt_path = _FROZEN_PROMPTS / "dft_structure.txt"
    if not prompt_path.is_file():
        pytest.skip("LLMSim prompt file not present in this checkout")
    clear_cache()

    result = llm_sim(
        json_pred=[{"bandgap": "2.5 eV"}],
        json_ref=[{"bandgap": "2.0 eV"}],
        prompt_path=str(prompt_path),
        judge_client=_always_match_judge,
    )
    assert result["num_match"] == 0
    assert result["verifier_revoked_count"] == 1
    assert result["f1"] == 0.0
    clear_cache()


def test_llm_sim_verifier_keeps_match_on_numeric_agreement():
    """Gemini says match, numerics agree → verifier preserves.

    ref = [{"bandgap": "2.0 eV"}], pred = [{"bandgap": "2.0 eV"}]
    rel_error = 0 → keep.
    Expected: num_match=1, verifier_revoked_count=0, f1=1.0.
    """
    from curie_rlm_env.judge_cache import clear_cache

    prompt_path = _FROZEN_PROMPTS / "dft_structure.txt"
    if not prompt_path.is_file():
        pytest.skip("LLMSim prompt file not present in this checkout")
    clear_cache()

    result = llm_sim(
        json_pred=[{"bandgap": "2.0 eV"}],
        json_ref=[{"bandgap": "2.0 eV"}],
        prompt_path=str(prompt_path),
        judge_client=_always_match_judge,
    )
    assert result["num_match"] == 1
    assert result["verifier_revoked_count"] == 0
    assert result["f1"] == 1.0
    clear_cache()


def test_scorers_llm_sim_does_not_have_repair_branch():
    """Defensive: source must not contain the deleted JSON-repair regex pattern."""
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "curie_rlm_env" / "scorers.py"
    ).read_text()
    assert 're.finditer(r",\\s*\\{"' not in src, "scorers.py still has the JSON-repair regex"
    assert "= json5.loads(output[: inds[-1]] + \"]\")" not in src
