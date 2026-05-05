"""Stage 3.5 — Baseline eval module tests.

Tests target the pure aggregation/IO functions; full live-eval requires
Qwen + Gemini endpoints (CLI invocation per scripts/run_baseline_eval.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from curie_rlm_env.baseline_eval import (
    TASK_IDS,
    aggregate_rollouts,
    write_baseline_output,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BASELINE_SRC = _PROJECT_ROOT / "src" / "curie_rlm_env" / "baseline_eval.py"


def _mk_rollout(task_id: str, record_id: str, reward: float) -> dict:
    return {
        "task_id": task_id,
        "record_id": record_id,
        "reward": reward,
        "headline_score": reward,
        "auxiliary_scores": {"rouge_lsum": 0.0, "bert_f1": 0.0},
        "tool_call_count": 0,
        "num_turns": 1,
        "completion_length": 100,
    }


# ---------------------------------------------------------------------------

def test_baseline_module_imports():
    # Sanity import — module must load without live LLM endpoints
    from curie_rlm_env import baseline_eval as be
    assert callable(be.aggregate_rollouts)
    assert callable(be.write_baseline_output)
    assert callable(be.run_baseline)


def test_baseline_writes_expected_schema(tmp_path):
    # Single-record aggregate → JSON schema matches the spec
    rollouts = [_mk_rollout("DFT-S", "rec_0", 0.7)]
    aggregated = aggregate_rollouts(rollouts, model="Qwen3.5-7B", split="test")
    out_path = tmp_path / "baseline.json"
    write_baseline_output(aggregated, out_path)

    parsed = json.loads(out_path.read_text())
    assert parsed["model"] == "Qwen3.5-7B"
    assert parsed["split"] == "test"
    assert parsed["n_problems"] == 1
    assert "per_task" in parsed
    assert "overall" in parsed
    assert "rollouts" in parsed
    assert parsed["per_task"]["DFT-S"]["mean_reward"] == 0.7
    assert parsed["per_task"]["DFT-S"]["n"] == 1
    assert "std" in parsed["per_task"]["DFT-S"]


def test_baseline_per_task_keys_match_10():
    # Aggregated per_task must have exactly the 10 canonical task IDs
    rollouts = [_mk_rollout("DFT-S", "rec_0", 0.5)]
    aggregated = aggregate_rollouts(rollouts, model="Qwen3.5-7B")
    expected = set(TASK_IDS)
    assert set(aggregated["per_task"].keys()) == expected
    assert len(expected) == 10


def test_baseline_handles_zero_reward(tmp_path):
    # Zero-reward rollout (e.g. failed inference) → JSON schema still valid
    rollouts = [_mk_rollout("BIOGR", "failed_rec", 0.0)]
    aggregated = aggregate_rollouts(rollouts, model="Qwen3.5-7B")
    out_path = tmp_path / "baseline_zero.json"
    write_baseline_output(aggregated, out_path)

    parsed = json.loads(out_path.read_text())
    assert parsed["per_task"]["BIOGR"]["mean_reward"] == 0.0
    assert parsed["per_task"]["BIOGR"]["n"] == 1
    assert parsed["overall"]["mean_reward"] == 0.0
    # All 10 tasks present even if 9 of them had zero rollouts
    for task_id in TASK_IDS:
        assert task_id in parsed["per_task"]


def test_baseline_default_split_is_test():
    # Stage 3.5: split is parametrized via --split flag; default must be "test".
    # Hardcoded train/val references are still forbidden in non-comment code.
    content = _BASELINE_SRC.read_text()
    code_lines = [
        line for line in content.splitlines()
        if not line.lstrip().startswith("#")
    ]
    code = "\n".join(code_lines)
    # Default param value should be "test"
    assert (
        'split: str = "test"' in code
        or 'split="test"' in code
        or "split='test'" in code
    ), "baseline_eval.py must default split parameter to 'test'"
    # No hardcoded train/val splits
    assert 'split="train"' not in code
    assert "split='train'" not in code
    assert 'split="val"' not in code
    assert "split='val'" not in code


# ---------------------------------------------------------------------------
# Strict: _run_one_rollout no longer swallows infrastructure failures.
# ---------------------------------------------------------------------------


def test_run_one_rollout_propagates_exceptions():
    """Strict default: a failed rollout must raise, not return reward=0."""
    import asyncio

    from curie_rlm_env.baseline_eval import _run_one_rollout

    class _FailingEnv:
        task_id = "DFT-C"
        dataset = [{"prompt": [{"role": "user", "content": "x"}], "answer": '{"r": 1}', "info": {}}]

        async def run_rollout(self, **_kwargs):
            raise RuntimeError("simulated infra failure (e.g. sandbox death)")

    sem = asyncio.Semaphore(1)
    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(_run_one_rollout(
            env=_FailingEnv(), example_idx=0, client=None,
            model="x", sampling_args={}, semaphore=sem,
        ))
    assert "simulated infra failure" in str(exc_info.value)


def test_run_one_rollout_does_not_inject_reward_zero_on_error():
    """Strict: the source must not contain a reward=0 record-emission catch."""
    src = _BASELINE_SRC.read_text()
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    # Strict: no `"reward": 0.0` shaped record produced from an except block
    # in the rollout function. The pre-fix code returned a fully-shaped dict
    # with reward=0.0 from `except Exception as e:`. Verify the construct is gone.
    assert "except Exception as e" not in code, (
        "baseline_eval._run_one_rollout must not catch broad Exception"
    )
