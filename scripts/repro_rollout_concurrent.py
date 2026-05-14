"""Stage 3a — does concurrency break state propagation in the rollout loop?

Production runs up to 64 concurrent rollouts inside ONE env_worker subprocess
(batch_size=8 × rollouts_per_example=4 × max_async_level=2). Our serial repro
(repro_rollout_phase.py) proved state propagation works for ONE rollout at a
time. This script asks: does it still work when N rollouts are in flight at
once on the same env instance?

If state propagation breaks here → bug is concurrency in the rollout loop.
If everything is healthy → bug is in the ZMQ subprocess boundary (Stage 3b).

Spawns N asyncio.gather() tasks each calling env.rollout() against the real
vLLM. Same iteration probes as before, labeled per-rollout. The id(state)
printed in each iteration tells us if any two rollouts ever see the same dict
(they shouldn't — each gets its own state via init_state).

Env vars:
  INFERENCE_SERVER_IP   host[:port] of running vLLM (default 127.0.0.1:8000)
  VLLM_API_KEY          (optional, default 'EMPTY')
  REPRO_MODEL           default Qwen/Qwen3-8B
  REPRO_NUM_ROLLOUTS    concurrent rollouts (default 16; bump toward 64 to
                        match the production max_async_level × rollouts_per_example)

Run:
    INFERENCE_SERVER_IP=127.0.0.1:8000 uv run python scripts/repro_rollout_concurrent.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from curie_rlm_env import CurieRLMEnv
from verifiers.types import ClientConfig


def vllm_client_config() -> ClientConfig:
    raw = os.environ.get("INFERENCE_SERVER_IP", "127.0.0.1:8000")
    if "://" not in raw:
        raw = f"http://{raw}"
    base_url = raw.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")
    return ClientConfig(
        client_type="openai_chat_completions",
        api_base_url=base_url,
        api_key_var="VLLM_API_KEY",
        reasoning_parser="deepseek_r1",
    )


def install_concurrent_probe(env: CurieRLMEnv) -> dict[str, Any]:
    """Single shared probe that logs every is_completed call across ALL rollouts.

    The shared probe is the whole point: in production all rollouts share one
    env instance. If concurrency is the bug, two rollouts' is_completed calls
    can interleave with one's submit_answer dispatch, and we'll see weirdness
    in the id(state) / rollout_id pairings.
    """
    orig = env.is_completed
    # Per-rollout counters keyed by rollout_id (or None for pre-setup state)
    per_rollout: dict[str, dict[str, Any]] = {}

    async def probed(state, **kwargs):
        rid = state.get("rollout_id")
        # Track per-rollout iteration count + state-id history
        info = per_rollout.setdefault(
            str(rid), {"iter": 0, "state_ids_seen": set()}
        )
        info["iter"] += 1
        info["state_ids_seen"].add(id(state))
        i = info["iter"]
        same_id_seen_before = (id(state) in info["state_ids_seen"]
                               and len(info["state_ids_seen"]) > 1)

        print(
            f"[{rid!s:>15} ITER {i:>2}] BEFORE | "
            f"id_state={id(state)} "
            f"final_answer_in_state={'final_answer' in state} "
            f"_curie_submit_calls={state.get('_curie_submit_answer_calls', 0)} "
            f"root_llm_turns={state.get('root_llm_turns', 0)} "
            f"trajectory_len={len(state.get('trajectory', []))} "
            f"final_env_response_set={state.get('final_env_response') is not None} "
            + (f"!!STATE_ID_CHANGED!! " if same_id_seen_before else "")
        )
        result = await orig(state, **kwargs)
        print(
            f"[{rid!s:>15} ITER {i:>2}] AFTER  | "
            f"→ {result} stop_condition={state.get('stop_condition')!r}"
        )
        return result

    env.is_completed = probed  # type: ignore[method-assign]
    return per_rollout


async def run_one(env: CurieRLMEnv, client_cfg: ClientConfig, example: dict, model: str, idx: int) -> dict[str, Any]:
    try:
        state = await env.rollout(
            input=example,
            client=client_cfg,
            model=model,
            sampling_args={
                "temperature": 1.0,
                "max_completion_tokens": 4096,
                "extra_body": {"tool_choice": "required"},  # Issue B fix
            },
        )
        return {
            "idx": idx,
            "rollout_id": state.get("rollout_id"),
            "stop_condition": state.get("stop_condition"),
            "final_answer_in_state": "final_answer" in state,
            "_curie_submit_answer_calls": state.get("_curie_submit_answer_calls", 0),
            "root_llm_turns": state.get("root_llm_turns", 0),
            "trajectory_len": len(state.get("trajectory", [])),
        }
    except Exception as exc:
        return {
            "idx": idx,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def main() -> None:
    print("=== Stage 3a — concurrent rollout-phase repro ===")
    client_cfg = vllm_client_config()
    print(f"vLLM base_url: {client_cfg.api_base_url}")
    model = os.environ.get("REPRO_MODEL", "Qwen/Qwen3-8B")
    num = int(os.environ.get("REPRO_NUM_ROLLOUTS", "16"))
    print(f"model:           {model}")
    print(f"concurrent N:    {num}")
    print()

    print("Constructing real CurieRLMEnv(continual_phase=1, split='train') …")
    env = CurieRLMEnv(continual_phase=1, split="train")
    ds = env.get_dataset()
    print(f"  → dataset has {len(ds)} examples")
    print(f"  → env.max_turns = {env.max_turns}")
    print()

    install_concurrent_probe(env)

    examples = [ds[i % len(ds)] for i in range(num)]
    print(f"Launching {num} concurrent rollouts via asyncio.gather …\n")

    summaries = await asyncio.gather(
        *[run_one(env, client_cfg, examples[i], model, i + 1) for i in range(num)],
        return_exceptions=False,
    )

    print("\n\n=== AGGREGATE ===")
    n_total = len(summaries)
    n_err = sum(1 for s in summaries if "error" in s)
    n_submit = sum(1 for s in summaries if s.get("_curie_submit_answer_calls", 0) > 0)
    n_final = sum(1 for s in summaries if s.get("final_answer_in_state"))
    n_answer_ready = sum(1 for s in summaries if s.get("stop_condition") == "answer_ready")

    print(f"  concurrent rollouts:    {n_total}")
    print(f"  errored:                {n_err}")
    print(f"  submit_answer called:   {n_submit}")
    print(f"  final_answer in state:  {n_final}")
    print(f"  stopped via answer_ready: {n_answer_ready}")
    print()

    # Stop-condition breakdown — most useful aggregate signal
    stops: dict[str, int] = {}
    for s in summaries:
        key = s.get("stop_condition") if "error" not in s else f"ERROR:{s['error'][:40]}"
        stops[str(key)] = stops.get(str(key), 0) + 1
    print("Stop conditions:")
    for cond, count in sorted(stops.items(), key=lambda kv: -kv[1]):
        print(f"  {cond}: {count}")

    print("\nPer-rollout breakdown:")
    for s in summaries:
        if "error" in s:
            print(f"  R{s['idx']}: ERROR {s['error']}")
        else:
            print(
                f"  R{s['idx']}: rollout_id={s['rollout_id']!r} "
                f"stop={s['stop_condition']!r} "
                f"submit={s['_curie_submit_answer_calls']} "
                f"final={s['final_answer_in_state']} "
                f"turns={s['root_llm_turns']}"
            )

    print("\n--- VERDICT ---")
    if n_submit > 0 and n_final < n_submit:
        ratio = n_final / n_submit
        print(f"BUG REPRODUCED UNDER CONCURRENCY:")
        print(f"  {n_submit} rollouts ran submit_answer, only {n_final} ended with final_answer in state.")
        print(f"  ({ratio:.0%} success rate; production showed ~0%)")
        print("  → State propagation breaks under concurrent load.")
        print("  → Iteration logs above show !!STATE_ID_CHANGED!! tags where")
        print("    a rollout's state object was replaced mid-flight, or rollout")
        print("    interleavings where one rollout's submit affected another's state.")
        sys.exit(1)
    elif n_submit == 0:
        print("INCONCLUSIVE: no rollout ever called submit_answer.")
        print("  → Check the no_tools_called stop frequency. With tool_choice='required'")
        print("    in extra_body, every turn should be a tool call. If submit_answer")
        print("    is never picked, the model may need more turns or different prompts.")
        sys.exit(2)
    elif n_final == n_submit:
        print(f"HEALTHY UNDER CONCURRENCY: {n_final}/{n_submit} terminated correctly.")
        print("  → Concurrency in the rollout loop is NOT the cause.")
        print("  → Bug requires the ZMQ env_worker subprocess boundary (Stage 3b).")
        sys.exit(0)
    else:
        print(f"PARTIAL: {n_final}/{n_submit} terminated correctly.")
        print("  → Intermittent. Re-run with higher REPRO_NUM_ROLLOUTS to see if it scales.")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())
