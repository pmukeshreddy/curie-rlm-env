"""
Stage 2 v3 — CurieRLMEnv integration tests (18 tests).

Guard #1 — Schema validation (curie_rlm_env/schema.py)
Guard #2 — Sub-LM oracle-leak interception via llm_batch
    Source: rlm_env.py:2611-2612 (llm_batch.__name__ = "llm_batch"; tools.append(llm_batch))
Guard #3 — Safeguards values flow to RLMEnv instance attributes
    Source: rlm_env.py:2403, 2404, 2415, 2416, 2455, 2458
Guard #4 — RLMMonitorRubric auto-attached
    Source: rlm_env.py:2517 (self.add_rubric(RLMMonitorRubric(...)))

Placeholder rubric: vf.Rubric() with no reward funcs (rubric.py:40 self.funcs = funcs or []).
Stage 3 replaces with CurieRubric dispatcher.
"""
import asyncio

import datasets
import pytest
import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMEnv, RLMMonitorRubric

from curie_rlm_env import CurieRLMEnv, load_curie_task, load_environment, validate_answer


@pytest.fixture(scope="module")
def env() -> CurieRLMEnv:
    # C1: load_environment(task_id: str, split: str = "test") -> CurieRLMEnv. No kwarg overrides.
    return load_environment("DFT-S", "test")


# -------- Guard #1 — schema validation (4 tests) -----------------------------

def test_schema_rejects_empty_string():
    with pytest.raises(ValueError):
        validate_answer("")

def test_schema_rejects_whitespace_only():
    with pytest.raises(ValueError):
        validate_answer("   ")

def test_schema_rejects_non_string():
    with pytest.raises(ValueError):
        validate_answer(42)

def test_schema_accepts_valid_content():
    # M6: validator returns None on success
    assert validate_answer("The answer is 42.") is None


def test_submit_answer_uses_worker_answer_contract():
    """Quote from rlm_env.py: 'with open(ANSWER_FILE, "w", encoding="utf-8") as f:'."""
    env = CurieRLMEnv.__new__(CurieRLMEnv)
    calls = []

    async def fake_execute_code(code, state_arg):
        calls.append((code, state_arg))
        return {"answer": {"ready": True, "content": "actual final answer"}}

    env._execute_code = fake_execute_code
    state = {"rollout_id": "rlm_unit"}

    result = asyncio.run(env.submit_answer("actual final answer", state))

    assert result == "Final answer submitted."
    assert state["final_answer"] == "actual final answer"
    assert calls == [
        (
            "answer['content'] = 'actual final answer'\nanswer['ready'] = True",
            state,
        )
    ]


def test_submit_answer_rejects_worker_mismatch():
    """Quote from rlm_env.py: 'if answer_ready: state["final_answer"] = answer.get("content", "")'."""
    env = CurieRLMEnv.__new__(CurieRLMEnv)

    async def fake_execute_code(code, state_arg):
        return {"answer": {"ready": False, "content": ""}}

    env._execute_code = fake_execute_code

    with pytest.raises(RuntimeError):
        asyncio.run(env.submit_answer("actual final answer", {"rollout_id": "rlm_unit"}))


def test_submit_answer_injects_state_arg():
    env = CurieRLMEnv.__new__(CurieRLMEnv)
    state = {"rollout_id": "rlm_unit"}

    updated = env.update_tool_args("submit_answer", {"content": "x"}, [], state)

    assert updated == {"content": "x", "state": state}


# -------- Guard #2 — llm_batch tool registration (1 test) --------------------

def test_llm_batch_tool_registered(env):
    # rlm_env.py:2611-2612: llm_batch.__name__ = "llm_batch"; tools.append(llm_batch)
    # The local `tools` in _build_fixed_root_tools() becomes self.root_tools, not self.tools.
    # rlm_env.py:2495: self.root_tool_names = [_tool_display_name(tool) for tool in self.root_tools]
    # env.tools (StatefulToolEnv) holds regular tools (e.g., call_bash_repl); llm_batch is in root_tools.
    assert "llm_batch" in [t.__name__ for t in env.root_tools]


# -------- Guard #3 — safeguards flow (6 tests) -------------------------------

def test_sub_llm_max_turns_flows_from_safeguards(env):
    # rlm_env.py:2403: self.sub_llm_max_turns = sub_llm_max_turns
    assert env.sub_llm_max_turns == 1

def test_sub_max_completion_tokens_flows_from_safeguards(env):
    # rlm_env.py:2404: self.sub_max_completion_tokens = sub_max_completion_tokens
    assert env.sub_max_completion_tokens == 8192

def test_sandbox_timeout_minutes_flows_from_safeguards(env):
    # rlm_env.py:2458: self.sandbox_timeout_minutes = sandbox_timeout_minutes
    # M4: 1 minute (preserves original 60s)
    assert env.sandbox_timeout_minutes == 1

def test_sandbox_memory_gb_flows_from_safeguards(env):
    # rlm_env.py:2455: self.sandbox_memory_gb = sandbox_memory_gb
    assert env.sandbox_memory_gb == 4

def test_code_execution_timeout_flows_from_safeguards(env):
    # rlm_env.py:2415: self.code_execution_timeout = code_execution_timeout
    assert env.code_execution_timeout == 120

def test_abort_on_code_timeout_flows_from_safeguards(env):
    # rlm_env.py:2416: self.abort_on_code_timeout = abort_on_code_timeout
    assert env.abort_on_code_timeout is True


# -------- Guard #4 — RLMMonitorRubric auto-attached (1 test) -----------------

def test_rlm_monitor_rubric_in_rubric_chain(env):
    # rlm_env.py:2517: self.add_rubric(RLMMonitorRubric(root_tool_names=self.root_tool_names))
    # environment.py:1233-1239: add_rubric wraps into RubricGroup on 2nd add
    rubrics = env.rubric.rubrics if isinstance(env.rubric, vf.RubricGroup) else [env.rubric]
    assert any(isinstance(r, RLMMonitorRubric) for r in rubrics)


# Placeholder rubric test removed in Stage 3b: CurieRubric (7 reward funcs) now
# replaces the empty vf.Rubric() placeholder. Stage 3 test_curie_rubric_replaces_placeholder
# (in tests/test_stage3_rubric.py) is the canonical validator of the new state.


# -------- Structural (4 tests) ----------------------------------------------

def test_curie_rlm_env_inherits_rlm_env(env):
    # CurieRLMEnv must inherit from verifiers.envs.experimental.rlm_env.RLMEnv
    assert isinstance(env, RLMEnv)

def test_load_environment_callable():
    # C1: public factory function
    assert callable(load_environment)

def test_load_curie_task_returns_dataset():
    # types.py:21: Dataset is from datasets package
    # data/curie/data/data/dft/inputs/ has 74 records (verified on disk)
    ds = load_curie_task("DFT-S", "test")
    assert isinstance(ds, datasets.Dataset)
    assert len(ds) > 0

def test_load_curie_task_invalid_task_id_raises():
    # M5: single ValueError class, never tuple
    with pytest.raises(ValueError):
        load_curie_task("BOGUS", "test")


# -------- Split validation (1 test) -----------------------------------------

def test_split_invalid_raises_value_error():
    # Stage 3.5: train/val/test are now all valid splits (read from
    # data/curie/splits/{split}.jsonl). Only invalid split names raise.
    with pytest.raises(ValueError):
        load_curie_task("DFT-S", "bogus")
