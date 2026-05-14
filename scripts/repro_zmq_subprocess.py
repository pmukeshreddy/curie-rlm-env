"""Stage 3b — does Issue A reproduce across the env_worker ZMQ subprocess boundary?

Stage 3a (in-process concurrent rollouts) was HEALTHY. The remaining suspect is
the boundary that prime-rl uses in production: a separate process running
verifiers.serve.server.env_server.EnvServer with rollout requests delivered via
ZMQ. This script spawns exactly that — ZMQEnvServer in a multiprocessing.spawn
subprocess — and sends N concurrent run_rollout requests via ZMQEnvClient.

The CurieRLMEnv probes added to env.py (id_state + state_keys at
is_completed/update_tool_args/submit_answer/answer_schema_valid) print to stderr
inside the worker subprocess. multiprocessing.spawn inherits stderr by default,
so [CURIE-DEBUG] lines appear in this script's stderr.

What this isolates:
  - In-process Stage 3a: HEALTHY 16/16 → not the rollout loop
  - Stage 3b (this script): tests env_worker subprocess + ZMQ msgpack round trip
    + worker's asyncio.create_task per request
  - If BUG-REPRODUCED here → root cause lives in the verifiers env_worker /
    ZMQ pipeline (state copy at the worker boundary, msgpack lossy round-trip,
    or concurrent task interleaving on the worker event loop).
  - If HEALTHY here → bug requires prime-rl orchestrator's specific batching:
    run_group, online_difficulty_filtering, max_async_level, DAPO buffer.

Cost: zero GPU on this script's side. Worker is CPU-only. vLLM is whatever you
already have running ($INFERENCE_SERVER_IP).

Env:
  INFERENCE_SERVER_IP    host[:port] of the vLLM (default 127.0.0.1:8000)
  VLLM_API_KEY           (optional, default 'EMPTY')
  REPRO_MODEL            default Qwen/Qwen3-8B
  REPRO_NUM_ROLLOUTS     concurrent rollouts via run_rollout (default 16)
  REPRO_LOG_DIR          where env_server / env_worker write log files
                         (default ./repro_zmq_logs)
  REPRO_ZMQ_ADDRESS      ZMQ bind address (default tcp://127.0.0.1:5555)

Run:
    INFERENCE_SERVER_IP=127.0.0.1:8000 \\
    uv run python scripts/repro_zmq_subprocess.py
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

from curie_rlm_env.continual import CONTINUAL_SEED, load_continual_phase_dataset
from verifiers.serve.client.zmq_env_client import ZMQEnvClient
from verifiers.serve.server.zmq_env_server import ZMQEnvServer
from verifiers.types import ClientConfig

ADDRESS = os.environ.get("REPRO_ZMQ_ADDRESS", "tcp://127.0.0.1:5555")
NUM_ROLLOUTS = int(os.environ.get("REPRO_NUM_ROLLOUTS", "16"))
MODEL = os.environ.get("REPRO_MODEL", "Qwen/Qwen3-8B")
LOG_DIR = Path(os.environ.get("REPRO_LOG_DIR", "./repro_zmq_logs")).resolve()


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


def server_target(address: str, log_dir: str) -> None:
    """Subprocess entrypoint — runs ZMQEnvServer.run_server which calls asyncio.run.

    Constructs CurieRLMEnv inside this subprocess via verifiers' env_router,
    matching the production env_worker boundary. CurieRLMEnv probes (id_state +
    state_keys) print to stderr from this subprocess and bubble up to the
    parent's stderr because multiprocessing.spawn inherits FDs.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ZMQEnvServer.run_server(
        env_id="curie-rlm-env",
        env_args={"continual_phase": 1, "split": "train", "seed": CONTINUAL_SEED},
        address=address,
        log_dir=log_dir,
        num_workers=1,
    )


async def run_one(client: ZMQEnvClient, example: dict, idx: int) -> dict[str, Any]:
    cfg = vllm_client_config()
    try:
        output = await client.run_rollout(
            input=example,
            client_config=cfg,
            model=MODEL,
            sampling_args={
                "temperature": 1.0,
                "max_completion_tokens": 4096,
                "extra_body": {"tool_choice": "required"},
            },
        )
        # RolloutOutput(dict) — top-level fields only. Custom state keys like
        # `_curie_submit_answer_calls` and `final_answer` are NOT here unless
        # state_columns explicitly named them. The orchestrator-relevant signals
        # are these standard fields:
        return {
            "idx": idx,
            "is_completed": output.get("is_completed", False),
            "stop_condition": output.get("stop_condition"),
            "reward": output.get("reward", 0.0),
            "answer_present": bool(output.get("answer")),
            "answer_len": len(output.get("answer") or ""),
            "error": output.get("error"),
        }
    except Exception as exc:
        return {"idx": idx, "error": f"{type(exc).__name__}: {exc}"}


async def amain() -> int:
    print("=== Stage 3b — env_server subprocess + ZMQ rollouts ===", flush=True)
    print(f"  ZMQ address:    {ADDRESS}", flush=True)
    print(f"  model:          {MODEL}", flush=True)
    print(f"  N rollouts:     {NUM_ROLLOUTS}", flush=True)
    print(f"  worker logs:    {LOG_DIR}", flush=True)
    print(flush=True)

    print("Loading dataset (main process — avoids reconstructing env here) ...", flush=True)
    ds = load_continual_phase_dataset(continual_phase=1, split="train", seed=CONTINUAL_SEED)
    examples = [ds[i % len(ds)] for i in range(NUM_ROLLOUTS)]
    print(f"  → {len(examples)} examples loaded\n", flush=True)

    # spawn (not fork) — Docker sandbox / asyncio context don't survive fork cleanly
    print("Spawning ZMQEnvServer subprocess (start_method=spawn) ...", flush=True)
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=server_target,
        args=(ADDRESS, str(LOG_DIR)),
        daemon=False,
    )
    proc.start()
    print(f"  → server PID: {proc.pid}\n", flush=True)

    verdict_code = 99
    client: ZMQEnvClient | None = None
    try:
        client = ZMQEnvClient(address=ADDRESS, name="repro-stage-3b")
        print("Waiting for env_server to become healthy (timeout 600s) ...", flush=True)
        await client.wait_for_server_startup(timeout=600.0)
        print("  → server is healthy\n", flush=True)

        print(f"Sending {NUM_ROLLOUTS} concurrent run_rollout requests over ZMQ ...\n", flush=True)
        results = await asyncio.gather(
            *[run_one(client, examples[i], i + 1) for i in range(NUM_ROLLOUTS)],
            return_exceptions=False,
        )

        print("\n\n=== AGGREGATE ===", flush=True)
        n_total = len(results)
        n_err = sum(1 for r in results if "error" in r)
        n_completed = sum(1 for r in results if r.get("is_completed"))
        n_answer_ready = sum(1 for r in results if r.get("stop_condition") == "answer_ready")
        rewards = [r.get("reward", 0.0) for r in results if "error" not in r]
        n_nonzero_reward = sum(1 for x in rewards if x and x != 0.0)
        n_zero_reward = sum(1 for x in rewards if x == 0.0)
        reward_min = min(rewards) if rewards else 0.0
        reward_max = max(rewards) if rewards else 0.0
        reward_mean = (sum(rewards) / len(rewards)) if rewards else 0.0

        print(f"  rollouts:                 {n_total}", flush=True)
        print(f"  errored:                  {n_err}", flush=True)
        print(f"  is_completed=True:        {n_completed}", flush=True)
        print(f"  stop=answer_ready:        {n_answer_ready}", flush=True)
        print(f"  reward != 0.0:            {n_nonzero_reward}", flush=True)
        print(f"  reward == 0.0:            {n_zero_reward}", flush=True)
        print(f"  reward min/mean/max:      {reward_min:.4f} / {reward_mean:.4f} / {reward_max:.4f}", flush=True)

        stops: dict[str, int] = {}
        for r in results:
            key = r.get("stop_condition") if "error" not in r else f"ERROR:{r['error'][:60]}"
            stops[str(key)] = stops.get(str(key), 0) + 1
        print("\nStop conditions:", flush=True)
        for cond, count in sorted(stops.items(), key=lambda kv: -kv[1]):
            print(f"  {cond}: {count}", flush=True)

        print("\nPer-rollout breakdown:", flush=True)
        for r in results:
            if "error" in r:
                print(f"  R{r['idx']:>2}: ERROR {r['error']}", flush=True)
            else:
                print(
                    f"  R{r['idx']:>2}: "
                    f"is_completed={r['is_completed']} "
                    f"stop={r['stop_condition']!r} "
                    f"reward={r['reward']:.4f} "
                    f"gt_answer_present={r['answer_present']} "
                    f"gt_answer_len={r['answer_len']}",
                    flush=True,
                )

        print("\n--- VERDICT ---", flush=True)
        if n_completed == 0:
            print("BROKEN: no rollout ever set is_completed=True.", flush=True)
            print(f"  → 0/{n_total} rollouts terminated via any @vf.stop predicate.", flush=True)
            print("  → Either rollouts hit max_turns silently, or RolloutOutput.is_completed", flush=True)
            print("    isn't being set by state_to_output despite the worker logs showing", flush=True)
            print("    is_completed=True. Inspect worker stderr to confirm.", flush=True)
            verdict_code = 1
        elif n_completed > 0 and n_answer_ready == 0:
            print(f"PARTIAL: {n_completed}/{n_total} completed, but ZERO via answer_ready.", flush=True)
            print("  → Rollouts terminate via other stops (no_tools_called, max_turns, error).", flush=True)
            print("    submit_answer never reached/succeeded. Issue B may be regressing or", flush=True)
            print("    the model is not invoking submit_answer in time.", flush=True)
            verdict_code = 2
        elif n_completed > 0 and n_zero_reward == n_completed:
            print(f"BUG REPRODUCED: {n_completed}/{n_total} rollouts terminated, ALL with reward=0.0.", flush=True)
            print("  → State propagation is healthy; rubric is returning zero on every rollout.", flush=True)
            print("  → DAPO online_difficulty_filtering would reject every group → trainer stuck.", flush=True)
            print("  → Next: dump state['final_answer'] vs state['answer'] (ground truth) for one", flush=True)
            print("    rollout from worker logs to see WHY rubric returns 0.", flush=True)
            verdict_code = 3
        else:
            print(f"HEALTHY at env_worker ZMQ subprocess boundary:", flush=True)
            print(f"  {n_answer_ready}/{n_total} stopped via answer_ready, {n_nonzero_reward} non-zero rewards", flush=True)
            print(f"  reward range [{reward_min:.4f}, {reward_max:.4f}], mean {reward_mean:.4f}", flush=True)
            print("  → The verifiers env_worker subprocess returns useful RolloutOutput.", flush=True)
            print("  → If production trainer is still stuck at step 0, the issue is in", flush=True)
            print("    prime-rl's filter/buffer (online_difficulty_filtering, zero_advantage,", flush=True)
            print("    max_async_level batching) — not in env or env_worker.", flush=True)
            verdict_code = 0

    finally:
        print("\nShutting down env_server subprocess ...", flush=True)
        if client is not None:
            try:
                await client.close()
            except Exception as exc:
                print(f"  client.close() raised: {exc}", flush=True)
        proc.terminate()
        proc.join(timeout=15)
        if proc.is_alive():
            print("  → SIGTERM ignored, sending SIGKILL", flush=True)
            proc.kill()
            proc.join(timeout=5)
        print(f"  → server exited (exitcode={proc.exitcode})", flush=True)

    return verdict_code


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
