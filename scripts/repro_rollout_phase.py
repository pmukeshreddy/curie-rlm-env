"""Reproduce the has_final_answer=False bug end-to-end against REAL vLLM.

No mocks. No GPU on this script itself — but it does call the production
vLLM server (Qwen3-8B on the training pod's GPU). Everything else is
production code path:

  - Real CurieRLMEnv (Stage 5 phase 1 free-form, same as training)
  - Real Docker sandbox (submit_answer's _execute_code is real)
  - Real verifiers rollout loop in multiturn_env
  - Real RLMEnv.env_response, real @vf.stop predicates, real active_rollouts
  - Real vLLM call via OpenAIChatCompletionsClient
  - Real Qwen3-8B Hermes tool-call parsing

What it skips:
  - prime-rl trainer (no weight updates, no checkpoints)
  - prime-rl orchestrator (no ZMQ env_worker, no DAPO filter, no batching)

That's the whole point: if has_final_answer=False shows up here too, the bug
is in the rollout loop or below. If everything is HEALTHY, the bug requires
the orchestrator/ZMQ layer.

Env vars required:
  INFERENCE_SERVER_IP   host[:port] of the running vLLM (e.g. 127.0.0.1:8000)
  VLLM_API_KEY          (optional, vLLM accepts any string; default 'EMPTY')

Optional:
  REPRO_MODEL           model id served by vLLM (default Qwen/Qwen3-8B)
  REPRO_NUM_ROLLOUTS    how many rollouts to run (default 4)
  REPRO_MAX_TURNS       override CurieRLMEnv max_turns (default 12)

Run:
    INFERENCE_SERVER_IP=127.0.0.1:8000 uv run python scripts/repro_rollout_phase.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from curie_rlm_env import CurieRLMEnv
from verifiers.types import ClientConfig


def vllm_client_config() -> ClientConfig:
    raw = os.environ.get("INFERENCE_SERVER_IP")
    if not raw:
        sys.exit(
            "ERROR: set INFERENCE_SERVER_IP to the vLLM host[:port]. "
            "Example: INFERENCE_SERVER_IP=127.0.0.1:8000"
        )
    if "://" not in raw:
        raw = f"http://{raw}"
    base_url = raw.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    # vLLM accepts any non-empty API key; the env var name is read by the
    # framework at request time (os.environ[api_key_var]).
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")
    return ClientConfig(
        client_type="openai_chat_completions",
        api_base_url=base_url,
        api_key_var="VLLM_API_KEY",
        reasoning_parser="deepseek_r1",  # same as production CLI flag
    )


def install_iteration_probe(env: CurieRLMEnv, label: str) -> None:
    """Log state across every rollout-loop iteration BEFORE/AFTER is_completed.

    is_completed runs at the head of each iteration; it sees state as left by
    the previous iteration's tool dispatch. That's exactly where the bug
    should surface — if submit_answer set state["final_answer"] last iteration
    and is_completed doesn't see it this iteration, the state propagation gap
    is concrete and reproducible.
    """
    orig = env.is_completed
    counter = {"n": 0}

    async def probed(state, **kwargs):
        counter["n"] += 1
        i = counter["n"]
        print(
            f"[{label} ITER {i:>2}] BEFORE is_completed | "
            f"id_state={id(state)} "
            f"rollout_id={state.get('rollout_id')!r} "
            f"final_answer_in_state={'final_answer' in state} "
            f"_curie_submit_calls={state.get('_curie_submit_answer_calls', 0)} "
            f"root_llm_turns={state.get('root_llm_turns', 0)} "
            f"trajectory_len={len(state.get('trajectory', []))} "
            f"final_env_response_set={state.get('final_env_response') is not None} "
            f"is_truncated={state.get('is_truncated', False)} "
            f"prompt_too_long={state.get('prompt_too_long', False)}"
        )
        result = await orig(state, **kwargs)
        print(
            f"[{label} ITER {i:>2}] AFTER  is_completed | "
            f"→ {result} stop_condition={state.get('stop_condition')!r}"
        )
        return result

    env.is_completed = probed  # type: ignore[method-assign]


async def run_one(env: CurieRLMEnv, client_cfg: ClientConfig, example: dict, model: str, label: str) -> dict[str, Any]:
    install_iteration_probe(env, label)

    state = await env.rollout(
        input=example,
        client=client_cfg,
        model=model,
        sampling_args={"temperature": 1.0, "max_completion_tokens": 4096},
    )

    summary = {
        "rollout_id": state.get("rollout_id"),
        "is_completed": state.get("is_completed"),
        "stop_condition": state.get("stop_condition"),
        "final_answer_in_state": "final_answer" in state,
        "final_answer_len": len(state["final_answer"]) if "final_answer" in state else 0,
        "_curie_submit_answer_calls": state.get("_curie_submit_answer_calls", 0),
        "root_llm_turns": state.get("root_llm_turns", 0),
        "sub_llm_call_count": state.get("sub_llm_call_count", 0),
        "trajectory_len": len(state.get("trajectory", [])),
        "is_truncated": state.get("is_truncated", False),
        "prompt_too_long": state.get("prompt_too_long", False),
    }
    return summary


async def main() -> None:
    print("=== Repro: real vLLM rollout phase ===")
    client_cfg = vllm_client_config()
    print(f"vLLM base_url: {client_cfg.api_base_url}")
    model = os.environ.get("REPRO_MODEL", "Qwen/Qwen3-8B")
    print(f"model:         {model}")
    num_rollouts = int(os.environ.get("REPRO_NUM_ROLLOUTS", "4"))
    print(f"num_rollouts:  {num_rollouts}")
    print()

    print("Constructing CurieRLMEnv(continual_phase=1, split='train') …")
    env = CurieRLMEnv(continual_phase=1, split="train")
    ds = env.get_dataset()
    print(f"  → dataset has {len(ds)} examples\n")

    if max_turns_override := os.environ.get("REPRO_MAX_TURNS"):
        env.max_turns = int(max_turns_override)
        print(f"  → overrode env.max_turns to {env.max_turns}\n")

    summaries: list[dict[str, Any]] = []
    for i in range(num_rollouts):
        example = ds[i % len(ds)]
        print(f"\n────── ROLLOUT {i+1}/{num_rollouts} (example task={example.get('task')!r}) ──────")
        try:
            summary = await run_one(env, client_cfg, example, model, label=f"R{i+1}")
        except Exception as exc:
            summary = {"error": f"{type(exc).__name__}: {exc}", "rollout_id": None}
            print(f"  EXCEPTION: {summary['error']}")
        summaries.append(summary)
        print(f"\nROLLOUT {i+1} SUMMARY: {summary}")

    print("\n\n=== AGGREGATE ===")
    n_total = len(summaries)
    n_with_final = sum(1 for s in summaries if s.get("final_answer_in_state"))
    n_submit_called = sum(1 for s in summaries if s.get("_curie_submit_answer_calls", 0) > 0)
    n_error = sum(1 for s in summaries if "error" in s)
    print(f"  rollouts:                 {n_total}")
    print(f"  errored:                  {n_error}")
    print(f"  submit_answer called ≥1:  {n_submit_called}")
    print(f"  final_answer in state:    {n_with_final}")
    print()
    print("Per-rollout breakdown:")
    for i, s in enumerate(summaries, 1):
        if "error" in s:
            print(f"  R{i}: ERROR {s['error']}")
        else:
            print(
                f"  R{i}: rollout_id={s['rollout_id']!r} "
                f"stop={s['stop_condition']!r} "
                f"submit_calls={s['_curie_submit_answer_calls']} "
                f"final_answer={s['final_answer_in_state']}"
            )

    print("\n--- VERDICT ---")
    if n_submit_called > 0 and n_with_final == 0:
        print("BUG REPRODUCED: submit_answer was called but no rollout ended with final_answer in state.")
        print("→ Iteration logs above show exactly when state[\"final_answer\"] disappears.")
        print("→ This proves the bug lives in the rollout loop / RLMEnv layer; NOT in")
        print("  the orchestrator or env_worker IPC.")
        sys.exit(1)
    elif n_submit_called == 0:
        print("INCONCLUSIVE: model never called submit_answer in any rollout.")
        print("→ Cold-start Qwen3-8B sometimes doesn't emit the tool. Run with")
        print("  REPRO_NUM_ROLLOUTS=16 to give it more samples.")
        sys.exit(2)
    elif n_with_final == n_submit_called:
        print(f"HEALTHY: {n_with_final}/{n_submit_called} rollouts that called submit_answer terminated with final_answer.")
        print("→ Bug does NOT reproduce in the standalone rollout loop.")
        print("→ The orchestrator / env_worker IPC is the suspect.")
        sys.exit(0)
    else:
        print(f"PARTIAL: {n_with_final}/{n_submit_called} rollouts terminated correctly.")
        print("→ Bug is intermittent in the rollout loop. Concurrency hypothesis still alive.")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())
