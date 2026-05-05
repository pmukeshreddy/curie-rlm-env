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

Training quote from src/curie_rlm_env/continual.py:
"Phase 2: 70% retrieval current tasks + 30% Phase 1 replay."
"""
from __future__ import annotations

from pathlib import Path

import yaml
import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import State

from .continual import CONTINUAL_SEED, load_continual_phase_dataset
from .datasets import load_curie_task
from .judge import make_gemini_judge_from_env
from .rubric import CurieRubric
from .schema import validate_answer

_CFG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "safeguards.yaml"
)


class CurieRLMEnv(RLMEnv):
    """CurieRLMEnv for continual training phases and single-task eval."""

    def __init__(
        self,
        task_id: str | None = None,
        split: str = "test",
        continual_phase: int | None = None,
        seed: int = CONTINUAL_SEED,
    ):
        if (task_id is None) == (continual_phase is None):
            raise ValueError(
                "CurieRLMEnv requires exactly one of task_id (single-task eval) "
                "or continual_phase (continual training)."
            )
        cfg = yaml.safe_load(_CFG_PATH.read_text())
        dataset = (
            load_continual_phase_dataset(continual_phase, split=split, seed=seed)
            if continual_phase is not None
            else load_curie_task(task_id, split)
        )
        judge_client = (
            make_gemini_judge_from_env()
            if continual_phase in {2, 3}
            else None
        )
        rubric = CurieRubric(judge_client=judge_client)
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
        self.task_id = task_id if task_id is not None else f"continual_phase_{continual_phase}"
        self.continual_phase = continual_phase
        self.seed = seed

    @vf.stop
    async def answer_schema_valid(self, state: State) -> bool:
        if "final_answer" not in state:
            return False
        validate_answer(state["final_answer"])
        return True


def load_task_environment(task_id: str, split: str = "test") -> CurieRLMEnv:
    """Load a single-task CURIE environment for eval and rubric compatibility."""
    return CurieRLMEnv(task_id=task_id, split=split)


def load_continual_environment(
    continual_phase: int,
    split: str = "train",
    seed: int = CONTINUAL_SEED,
) -> CurieRLMEnv:
    """Load a continual replay training environment."""
    return CurieRLMEnv(continual_phase=continual_phase, split=split, seed=seed)


def load_environment(
    task_id: str | None = None,
    split: str = "test",
    continual_phase: int | None = None,
    seed: int = CONTINUAL_SEED,
) -> CurieRLMEnv:
    """Prime/verifiers entrypoint.

    Training configs pass continual_phase=<1|2|3>. Baseline/eval callers keep task_id.
    """
    if continual_phase is not None:
        return load_continual_environment(continual_phase=continual_phase, split=split, seed=seed)
    if task_id is None:
        raise ValueError("load_environment requires task_id for eval or continual_phase for continual training.")
    return load_task_environment(task_id=task_id, split=split)
