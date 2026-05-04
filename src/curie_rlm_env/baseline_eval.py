"""Stage 3.5 — Baseline eval. Qwen3.5-7B + CurieRLMEnv on a held-out split.

Captures per-task floor numbers for Stage 7's delta computation. No GRPO,
no training — pure inference + scoring with the locked CurieRubric.

Stage 3.5 follow-up: split is parametrized (default "test"); accepts
"train"/"val"/"test" once data/curie/splits/{split}.jsonl exists (built by
scripts/build_splits.py).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Optional

import verifiers as vf

from .env import load_environment
from .rubric import CurieRubric

logger = logging.getLogger(__name__)

# Canonical 10-task list, mirrors config/curie_tasks.yaml
TASK_IDS: tuple[str, ...] = (
    "DFT-S", "DFT-P", "DFT-C", "MPVE",
    "BIOGR", "PDB",
    "HFE", "HFD", "QECC_65", "GEO",
)

# Tasks that require the LLMSim judge (gemini-2.5-pro)
RETRIEVAL_TASKS = frozenset({"DFT-S", "DFT-P", "MPVE"})


# ---------------------------------------------------------------------------
# Pure functions (testable without LLM endpoints)
# ---------------------------------------------------------------------------

def aggregate_rollouts(
    rollouts: list[dict],
    model: str,
    split: str = "test",
) -> dict[str, Any]:
    """Aggregate per-rollout records into the canonical output JSON structure."""
    per_task: dict[str, dict[str, float | int]] = {}
    for task_id in TASK_IDS:
        task_rollouts = [r for r in rollouts if r.get("task_id") == task_id]
        rewards = [float(r["reward"]) for r in task_rollouts]
        headlines = [
            float(r.get("headline_score", r["reward"])) for r in task_rollouts
        ]
        per_task[task_id] = {
            "mean_reward": statistics.mean(rewards) if rewards else 0.0,
            "mean_headline": statistics.mean(headlines) if headlines else 0.0,
            "std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            "n": len(rewards),
        }

    all_rewards = [float(r["reward"]) for r in rollouts]
    all_headlines = [float(r.get("headline_score", r["reward"])) for r in rollouts]
    overall = {
        "mean_reward": statistics.mean(all_rewards) if all_rewards else 0.0,
        "mean_headline": statistics.mean(all_headlines) if all_headlines else 0.0,
    }

    return {
        "model": model,
        "split": split,
        "n_problems": len(rollouts),
        "per_task": per_task,
        "overall": overall,
        "rollouts": rollouts,
    }


def write_baseline_output(aggregated: dict[str, Any], output_path: Path) -> None:
    """Write aggregated baseline JSON to disk. Creates parent dir if missing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregated, indent=2))


# ---------------------------------------------------------------------------
# Live-LLM functions (require Qwen vLLM endpoint + Gemini API key)
# ---------------------------------------------------------------------------

def _make_qwen_client(endpoint: str) -> "vf.Client":
    """Build a verifiers OpenAI-compatible client pointing at the Qwen vLLM endpoint."""
    from verifiers import OpenAIChatCompletionsClient
    return OpenAIChatCompletionsClient(
        base_url=endpoint,
        api_key=os.environ.get("QWEN_API_KEY", "EMPTY"),
    )


def _make_gemini_judge() -> Callable[[str], str]:
    """Build the gemini-2.5-pro judge callable for LLMSim tasks.

    Hard-fails if GEMINI_API_KEY env var is missing or google-genai is not
    installed (ZERO-FALLBACK: do not silently switch to a smaller judge).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY env var required for LLMSim judge (gemini-2.5-pro). "
            "Set it before running baseline eval."
        )
    try:
        from google import genai
    except ImportError as e:
        raise RuntimeError(
            "google-genai not installed. Add it to pyproject.toml deps "
            "before running baseline eval."
        ) from e

    client = genai.Client(api_key=api_key)

    def judge_call(prompt: str) -> str:
        response = client.models.generate_content(
            model="gemini-2.5-pro", contents=prompt
        )
        return response.text or "{}"

    return judge_call


def _attach_judge_to_env(env: Any, judge_callable: Callable[[str], str]) -> None:
    """Inject judge_client into the CurieRubric inside the env's rubric chain."""
    rubrics = (
        env.rubric.rubrics
        if isinstance(env.rubric, vf.RubricGroup)
        else [env.rubric]
    )
    for r in rubrics:
        if isinstance(r, CurieRubric):
            r._judge_client = judge_callable
            return
    raise RuntimeError(
        "CurieRubric not found in env.rubric chain — judge cannot be attached"
    )


def _extract_rollout_record(
    env: Any, example_idx: int, output: Any
) -> dict[str, Any]:
    """Extract per-rollout fields from a verifiers RolloutOutput."""
    state = getattr(output, "state", None) or {}
    info = env.dataset[example_idx].get("info", {}) or {}
    record_id = info.get("record_id", f"unknown_{example_idx}")
    metrics = state.get("metrics", {}) or {}
    completion = state.get("completion", "")
    completion_length = (
        len(completion)
        if isinstance(completion, str)
        else sum(len(m.get("content", "")) for m in completion if isinstance(m, dict))
    )
    trajectory = state.get("trajectory", []) or []
    tool_call_count = sum(
        1 for t in trajectory if isinstance(t, dict) and t.get("tool_call")
    )
    return {
        "task_id": env.task_id,
        "record_id": record_id,
        "reward": float(state.get("reward", 0.0)),
        "headline_score": float(state.get("reward", 0.0)),
        "auxiliary_scores": {
            "rouge_lsum": float(metrics.get("_aux_rouge_lsum", 0.0)),
            "bert_f1": float(metrics.get("_aux_bert_f1", 0.0)),
        },
        "tool_call_count": tool_call_count,
        "num_turns": len(trajectory),
        "completion_length": completion_length,
    }


async def _run_one_rollout(
    env: Any,
    example_idx: int,
    client: Any,
    model: str,
    sampling_args: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Run one rollout, capture metrics. On failure: log + return reward=0 record."""
    async with semaphore:
        info = env.dataset[example_idx].get("info", {}) or {}
        record_id = info.get("record_id", f"unknown_{example_idx}")
        try:
            input_dict = {
                "prompt": env.dataset[example_idx]["prompt"],
                "answer": env.dataset[example_idx].get("answer", ""),
                "task": env.task_id,
                "info": info,
                "example_id": example_idx,
            }
            output = await env.run_rollout(
                input=input_dict,
                client=client,
                model=model,
                sampling_args=sampling_args,
            )
            return _extract_rollout_record(env, example_idx, output)
        except Exception as e:
            logger.error(
                "Rollout failed for task=%s record_id=%s: %s",
                env.task_id, record_id, e,
            )
            return {
                "task_id": env.task_id,
                "record_id": record_id,
                "reward": 0.0,
                "headline_score": 0.0,
                "auxiliary_scores": {"rouge_lsum": 0.0, "bert_f1": 0.0},
                "tool_call_count": 0,
                "num_turns": 0,
                "completion_length": 0,
                "error": str(e),
            }


async def run_baseline(
    qwen_endpoint: str,
    output_path: Path | str,
    model: str = "Qwen/Qwen3.5-7B",
    max_concurrency: int = 4,
    sampling_args: Optional[dict[str, Any]] = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
    split: str = "test",
    sub_llm_max_turns_override: Optional[int] = None,
) -> dict[str, Any]:
    """Run the full baseline eval over all 10 tasks on the chosen split.

    Hard-fails at startup if the Qwen client or Gemini judge cannot be built.

    Stage 7: sub_llm_max_turns_override allows the RLM-on/off ablation to flip
    sub-LM recursion. None → use safeguards.yaml default. 0 → flat LM (no
    recursion). 1+ → RLM scaffold with that many sub-LM turns.
    """
    if sampling_args is None:
        sampling_args = {"temperature": 0.0, "max_tokens": 2048}

    # Hard-fail at startup (ZERO-FALLBACK): if endpoints unreachable, fail loud.
    client = _make_qwen_client(qwen_endpoint)
    judge = _make_gemini_judge()

    semaphore = asyncio.Semaphore(max_concurrency)
    rollouts: list[dict[str, Any]] = []

    for task_id in TASK_IDS:
        env = load_environment(task_id, split=split)
        if sub_llm_max_turns_override is not None:
            env.sub_llm_max_turns = sub_llm_max_turns_override
        if task_id in RETRIEVAL_TASKS:
            _attach_judge_to_env(env, judge)
        n = len(env.dataset)
        if progress is not None:
            progress(task_id, 0, n)
        coros = [
            _run_one_rollout(env, i, client, model, sampling_args, semaphore)
            for i in range(n)
        ]
        results = await asyncio.gather(*coros)
        if progress is not None:
            progress(task_id, n, n)
        rollouts.extend(results)

    aggregated = aggregate_rollouts(rollouts, model=model, split=split)
    write_baseline_output(aggregated, Path(output_path))
    return aggregated
