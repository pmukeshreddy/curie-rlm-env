"""CurieRLMEnv — Stage 2 wiring layer for CURIE benchmark.

Inherits verifiers.envs.experimental.rlm_env.RLMEnv. Reads safeguards from
config/safeguards.yaml and passes verbatim-named kwargs to super().__init__().

Stage 3b: vf.Rubric() placeholder replaced with CurieRubric() per-task
dispatcher. Note: CurieRubric judge_client defaults to None — production code
that needs LLMSim must construct CurieRubric directly with a real judge.

is_completed cannot be overridden (it is @final at environment.py:658). Schema
validation is wired via a @vf.stop-decorated method that returns True (signal
stop) or raises ValueError (loud schema fail). Returns False ONLY when the
final answer is not yet present in state — the multiturn-stop convention.
"""
from __future__ import annotations

from pathlib import Path

import yaml
import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import State

from .datasets import load_curie_task
from .rubric import CurieRubric
from .schema import validate_answer

_CFG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "safeguards.yaml"
)


class CurieRLMEnv(RLMEnv):
    """Single-task CurieRLMEnv. Safeguards from config/safeguards.yaml only."""

    def __init__(self, task_id: str, split: str = "test"):
        cfg = yaml.safe_load(_CFG_PATH.read_text())
        dataset = load_curie_task(task_id, split)
        rubric = CurieRubric()
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            sub_llm_max_turns=cfg["rlm_env"]["sub_llm_max_turns"],
            sub_max_completion_tokens=cfg["rlm_env"]["sub_max_completion_tokens"],
            sandbox_timeout_minutes=cfg["sandbox"]["sandbox_timeout_minutes"],
            sandbox_memory_gb=cfg["sandbox"]["sandbox_memory_gb"],
            code_execution_timeout=cfg["sandbox"]["code_execution_timeout"],
            abort_on_code_timeout=cfg["sandbox"]["abort_on_code_timeout"],
        )
        self.task_id = task_id

    @vf.stop
    async def answer_schema_valid(self, state: State) -> bool:
        if "final_answer" not in state:
            return False
        validate_answer(state["final_answer"])
        return True


def load_environment(task_id: str, split: str = "test") -> CurieRLMEnv:
    return CurieRLMEnv(task_id, split)
