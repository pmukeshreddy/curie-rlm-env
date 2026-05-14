"""Debug the rollout phase end-to-end without vLLM or training.

THIS IS WHAT'S MOCKED — and only this:
  env.get_model_response — returns a hardcoded ChatCompletion with a
  submit_answer tool call. We control the model's output so the rollout
  ALWAYS attempts submit_answer; otherwise cold-start Qwen3 picks
  call_python_repl half the time and we can't isolate the state bug.

THIS IS REAL — everything else, byte-for-byte production code:
  - CurieRLMEnv(continual_phase=1, split="train")
  - Real Docker sandbox via LocalDockerRLMExecutor
  - submit_answer running _execute_code on the sandbox
  - multiturn_env.rollout() — the actual loop the training uses
  - RLMEnv.env_response, RLMEnv.active_rollouts setup, _setup_interception_and_register
  - All @vf.stop predicates (answer_ready, answer_schema_valid, has_final_env_response,
    max_turns_reached, has_error, prompt_too_long, max_total_completion_tokens_reached)
  - All @vf.cleanup hooks
  - update_tool_args injecting state into submit_answer
  - The ZMQ-free in-process path (no env_worker, no orchestrator)

ONE command, no vLLM, no training, no GPU on the script itself (the sandbox
is CPU-only). Prints state at every iteration of the rollout loop so you can
see exactly when (or if) state["final_answer"] disappears.

Run:
    uv run python scripts/repro_rollout_phase.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import Function

from curie_rlm_env import CurieRLMEnv


def fake_submit_answer_response(answer_text: str) -> ChatCompletion:
    """Same shape Qwen3+Hermes emits in production — just no network call."""
    tool_call = ChatCompletionMessageToolCall(
        id="tc_repro_submit_1",
        type="function",
        function=Function(
            name="submit_answer",
            arguments=json.dumps({"content": answer_text}),
        ),
    )
    return ChatCompletion(
        id="repro_completion_1",
        object="chat.completion",
        created=0,
        model="repro-mock",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="",
                    tool_calls=[tool_call],
                ),
            )
        ],
    )


def install_probes(env: CurieRLMEnv, answer_text: str) -> dict[str, Any]:
    """Wrap is_completed (every loop iter) + get_model_response (mocked LLM).
    Returns a stats dict the caller reads after the rollout finishes.
    """
    stats: dict[str, Any] = {"iter": 0, "model_calls": 0}

    orig_is_completed = env.is_completed

    async def probed_is_completed(state, **kwargs):
        stats["iter"] += 1
        i = stats["iter"]
        print(
            f"[ITER {i:>2}] BEFORE is_completed | "
            f"id_state={id(state)} "
            f"rollout_id={state.get('rollout_id')!r} "
            f"final_answer_in_state={'final_answer' in state} "
            f"_curie_submit_calls={state.get('_curie_submit_answer_calls', 0)} "
            f"root_llm_turns={state.get('root_llm_turns', 0)} "
            f"trajectory_len={len(state.get('trajectory', []))} "
            f"final_env_response_set={state.get('final_env_response') is not None} "
            f"_rlm_stop_error={state.get('_rlm_stop_error') is not None}"
        )
        result = await orig_is_completed(state, **kwargs)
        print(
            f"[ITER {i:>2}] AFTER  is_completed | "
            f"→ {result} stop_condition={state.get('stop_condition')!r}"
        )
        return result

    async def fake_get_model_response(state, prompt_messages, **kwargs):
        stats["model_calls"] += 1
        print(f"[MODEL] call #{stats['model_calls']} (trajectory len={len(state.get('trajectory', []))})")
        return fake_submit_answer_response(answer_text)

    env.is_completed = probed_is_completed  # type: ignore[method-assign]
    env.get_model_response = fake_get_model_response  # type: ignore[method-assign]
    return stats


async def main() -> None:
    print("=== Rollout-phase repro ===")
    print("Constructing real CurieRLMEnv(continual_phase=1, split='train') …")
    env = CurieRLMEnv(continual_phase=1, split="train")
    ds = env.get_dataset()
    print(f"  → real dataset has {len(ds)} examples")
    print(f"  → real env.max_turns = {env.max_turns}")
    print(f"  → real sandbox client = {type(env._executor.sandbox_client).__name__}")
    print()

    stats = install_probes(env, answer_text="THE REPRO FINAL ANSWER")

    example = ds[0]
    print(f"Running ONE rollout against example task={example.get('task')!r}\n")

    state = await env.rollout(
        input=example,
        client=None,        # bypassed by our fake_get_model_response
        model="repro-mock",
        sampling_args=None,
    )

    print("\n=== FINAL STATE ===")
    print(f"  rollout_id:                 {state.get('rollout_id')!r}")
    print(f"  is_completed:               {state.get('is_completed')}")
    print(f"  stop_condition:             {state.get('stop_condition')!r}")
    print(f"  'final_answer' in state:    {'final_answer' in state}")
    if "final_answer" in state:
        fa = state["final_answer"]
        print(f"  final_answer (len={len(fa)}): {fa[:120]!r}{'…' if len(fa) > 120 else ''}")
    print(f"  _curie_submit_answer_calls: {state.get('_curie_submit_answer_calls', 'MISSING')}")
    print(f"  model calls made:           {stats['model_calls']}")
    print(f"  is_completed iterations:    {stats['iter']}")
    print(f"  trajectory steps:           {len(state.get('trajectory', []))}")
    print(f"  root_llm_turns:             {state.get('root_llm_turns', 0)}")
    print(f"  sub_llm_call_count:         {state.get('sub_llm_call_count', 0)}")
    print(f"  final_env_response:         {'SET' if state.get('final_env_response') else 'None'}")

    print("\n--- VERDICT ---")
    submit_calls = state.get("_curie_submit_answer_calls", 0)
    has_final = "final_answer" in state
    stop = state.get("stop_condition")

    if submit_calls > 0 and not has_final:
        print("BUG REPRODUCED:")
        print(f"  submit_answer ran {submit_calls} time(s) successfully,")
        print(f"  but state ended with NO 'final_answer'. Rollout terminated via")
        print(f"  stop_condition={stop!r} instead of answer_ready/answer_schema_valid.")
        print("\n  → State mutations from submit_answer are NOT propagating to the")
        print("    @vf.stop predicate state. Bug confirmed in the rollout loop layer.")
        print("    Look at the ITER lines above — compare _curie_submit_calls and")
        print("    final_answer_in_state across iterations. Whichever iter shows")
        print("    submit_calls go up but final_answer_in_state stays False is where")
        print("    the state mutation got dropped.")
        sys.exit(1)
    elif submit_calls == 0:
        print("UNEXPECTED:")
        print("  Mocked model never reached submit_answer (it should be EVERY call).")
        print("  Something in the framework's tool-dispatch chain rejected the call.")
        print("  Check the ITER logs for stop_condition or errors.")
        sys.exit(2)
    elif has_final and stop in {"answer_ready", "answer_schema_valid", "has_final_env_response"}:
        print(f"HEALTHY: rollout terminated via {stop!r} with final_answer set.")
        print(f"  submit_answer ran {submit_calls} time(s); state propagation works.")
        print("  → Bug does NOT reproduce in the standalone rollout loop.")
        print("  → Production bug requires the orchestrator/env_worker IPC layer or")
        print("    concurrent rollouts. Next: stress test with asyncio.gather(N=32+).")
        sys.exit(0)
    else:
        print(f"WEIRD: submit_calls={submit_calls} has_final={has_final} stop={stop!r}")
        print("  Read the ITER logs and tell me what happened.")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())
