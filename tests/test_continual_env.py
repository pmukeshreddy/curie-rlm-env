"""Continual training environment wiring tests.

Repository quotes anchoring these tests:
- src/curie_rlm_env/env.py: "Training configs pass continual_phase=<1|2|3>."
- src/curie_rlm_env/rubric.py: "LLMSim for {task_id} requires judge_client; got None"
"""
from __future__ import annotations

import datasets
import pytest
import verifiers as vf

import curie_rlm_env.env as env_mod
from curie_rlm_env import CurieRubric


def _mk_dataset(task_id: str = "DFT-S") -> datasets.Dataset:
    return datasets.Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": f"prompt {task_id}"}],
            "answer": "{}",
            "info": {
                "task_id": task_id,
                "record_id": f"{task_id}-0",
                "difficulty": "unit",
                "dft_field": None,
            },
        }
    ])


def _curie_rubric_from_env(env) -> CurieRubric:
    rubrics = env.rubric.rubrics if isinstance(env.rubric, vf.RubricGroup) else [env.rubric]
    for rubric in rubrics:
        if isinstance(rubric, CurieRubric):
            return rubric
    raise AssertionError("CurieRubric missing from environment rubric chain")


def test_continual_phase2_wires_gemini_judge(monkeypatch):
    sentinel = lambda prompt: '{"json_extracted_index": 0}'
    monkeypatch.setattr(
        env_mod,
        "load_continual_phase_dataset",
        lambda continual_phase, split, seed: _mk_dataset("DFT-S"),
    )
    monkeypatch.setattr(env_mod, "make_gemini_judge_from_env", lambda: sentinel)

    env = env_mod.load_environment(continual_phase=2, split="train")

    assert env.continual_phase == 2
    assert _curie_rubric_from_env(env)._judge_client is sentinel


def test_continual_phase3_wires_gemini_judge(monkeypatch):
    sentinel = lambda prompt: '{"json_extracted_index": 0}'
    monkeypatch.setattr(
        env_mod,
        "load_continual_phase_dataset",
        lambda continual_phase, split, seed: _mk_dataset("DFT-S"),
    )
    monkeypatch.setattr(env_mod, "make_gemini_judge_from_env", lambda: sentinel)

    env = env_mod.load_environment(continual_phase=3, split="train")

    assert env.continual_phase == 3
    assert _curie_rubric_from_env(env)._judge_client is sentinel


def test_continual_phase1_does_not_build_gemini_judge(monkeypatch):
    def fail_judge():
        raise AssertionError("Phase 1 must not construct a Gemini judge")

    monkeypatch.setattr(
        env_mod,
        "load_continual_phase_dataset",
        lambda continual_phase, split, seed: _mk_dataset("DFT-C"),
    )
    monkeypatch.setattr(env_mod, "make_gemini_judge_from_env", fail_judge)

    env = env_mod.load_environment(continual_phase=1, split="train")

    assert env.continual_phase == 1
    assert _curie_rubric_from_env(env)._judge_client is None


def test_phase_keyword_removed_from_public_training_entrypoint():
    with pytest.raises(TypeError):
        env_mod.load_environment(phase=2, split="train")
