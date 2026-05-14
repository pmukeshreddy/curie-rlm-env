"""Reproduce the has_final_answer=False bug by running ONE real CurieRLMEnv
rollout end-to-end on a single host. No GPU, no vLLM, no orchestrator, no
training. The model is replaced by a fake that always emits a `submit_answer`
tool call; everything else (sandbox, RLMEnv loop, @vf.stop predicates,
update_tool_args injection, env_response, active_rollouts) is production code.

What it produces:
  - prints state at each iteration of the real rollout loop
  - shows whether `has_final_answer` ever flips True
  - shows the stop_condition that actually terminates the rollout

Requirements on the host that runs this:
  - Docker daemon reachable (the sandbox is real; only the LLM is mocked)
  - Curie dataset present (data/curie LFS submodule fetched)
  - The same uv environment used for training

Run with:
    uv run python scripts/repro_rollout_phase.py
"""
from __future__ import annotations

import asyncio
import json
import sys

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import Function

from curie_rlm_env import CurieRLMEnv


# --- Fake model response (real OpenAI types, just no network) ---------------

def make_submit_answer_response(answer_text: str) -> ChatCompletion:
    """A real ChatCompletion object the framework can parse, carrying ONE
    submit_answer tool call. Same shape Qwen3 would emit through Hermes."""
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


# --- Per-iteration probe wrapper --------------------------------------------

def install_iteration_probe(env: CurieRLMEnv) -> None:
    """Wrap is_completed so we can log state at every loop iteration.

    `is_completed` is the first thing the rollout loop calls each cycle, so it
    sees state as it ENTERS the iteration — pre-tool-dispatch, pre-model. That
    means it observes whatever the previous iteration's tool dispatch left
    behind in state.
    """
    orig_is_completed = env.is_completed
    iteration = {"n": 0}

    async def probed(state, **kwargs):
        iteration["n"] += 1
        rid = state.get("rollout_id")
        print(
            f"[ITER {iteration['n']:>2}] BEFORE is_completed | "
            f"id_state={id(state)} "
            f"rollout_id={rid!r} "
            f"final_answer_in_state={'final_answer' in state} "
            f"_curie_submit_calls={state.get('_curie_submit_answer_calls', 0)} "
            f"trajectory_len={len(state.get('trajectory', []))} "
            f"final_env_response_set={state.get('final_env_response') is not None}"
        )
        result = await orig_is_completed(state, **kwargs)
        print(
            f"[ITER {iteration['n']:>2}] AFTER  is_completed | "
            f"→ {result} stop_condition={state.get('stop_condition')!r}"
        )
        return result

    env.is_completed = probed  # type: ignore[method-assign]


# --- Bypass the real LLM client ---------------------------------------------

def install_fake_model_response(env: CurieRLMEnv, answer_text: str) -> None:
    """Replace env.get_model_response so every model call returns the same
    submit_answer tool call. Bypasses the entire async OpenAI client wrapping."""

    async def fake_get_model_response(state, prompt_messages, **kwargs):
        print(f"[MODEL] fake get_model_response called (trajectory len={len(state.get('trajectory', []))})")
        return make_submit_answer_response(answer_text)

    env.get_model_response = fake_get_model_response  # type: ignore[method-assign]


# --- Main -------------------------------------------------------------------

async def main() -> None:
    print("Constructing CurieRLMEnv(continual_phase=1, split='train') …")
    env = CurieRLMEnv(continual_phase=1, split="train")
    print(f"  → env ready, dataset has {len(env.get_dataset())} examples")

    install_iteration_probe(env)
    install_fake_model_response(env, answer_text="The canonical repro answer about HFE.")

    # First example from the dataset; rollout loop will pull prompts from it.
    example = env.get_dataset()[0]
    print(f"  → using example task={example.get('task')!r}")
    print()

    print("Calling env.rollout(...) — this drives the real rollout loop\n")
    state = await env.rollout(
        input=example,
        client=None,            # not used (get_model_response is bypassed)
        model="repro-mock",
        sampling_args=None,
    )

    print("\n=== FINAL STATE ===")
    print(f"  rollout_id:                 {state.get('rollout_id')!r}")
    print(f"  is_completed:               {state.get('is_completed')}")
    print(f"  stop_condition:             {state.get('stop_condition')!r}")
    print(f"  'final_answer' in state:    {'final_answer' in state}")
    print(f"  final_answer:               {state.get('final_answer', '<MISSING>')!r}")
    print(f"  _curie_submit_answer_calls: {state.get('_curie_submit_answer_calls', 'MISSING')}")
    print(f"  trajectory steps:           {len(state.get('trajectory', []))}")
    print(f"  root_llm_turns:             {state.get('root_llm_turns', 0)}")
    print(f"  sub_llm_call_count:         {state.get('sub_llm_call_count', 0)}")
    print(f"  final_env_response:         {'SET' if state.get('final_env_response') else 'None'}")

    print("\n--- VERDICT ---")
    if "final_answer" in state and state.get("stop_condition") in {"answer_ready", "answer_schema_valid", "has_final_env_response"}:
        print("HEALTHY: rollout terminated via final-answer path.")
        print("  → Bug does not reproduce in isolated rollout. Suspect: env_worker IPC")
        print("  → or concurrent rollouts. Next: stress with asyncio.gather(N=64).")
        sys.exit(0)
    else:
        print("BUG REPRODUCED:")
        print(f"  - stop_condition='{state.get('stop_condition')}' (expected 'answer_ready' or similar)")
        print(f"  - final_answer in state? {'final_answer' in state}")
        print(f"  - _curie_submit_answer_calls={state.get('_curie_submit_answer_calls', 0)}"
              " (>0 means submit_answer ran; state still shows no final_answer → STATE LOSS)")
        print("\nIteration log above shows where state['final_answer'] failed to propagate.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
