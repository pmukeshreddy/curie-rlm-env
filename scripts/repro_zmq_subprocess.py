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
        state = getattr(output, "state", {}) or {}
        if not isinstance(state, dict):
            state = {}
        return {
            "idx": idx,
            "rollout_id": state.get("rollout_id"),
            "stop_condition": state.get("stop_condition"),
            "final_answer_in_state": "final_answer" in state,
            "_curie_submit_answer_calls": state.get("_curie_submit_answer_calls", 0),
            "root_llm_turns": state.get("root_llm_turns", 0),
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
        n_submit = sum(1 for r in results if r.get("_curie_submit_answer_calls", 0) > 0)
        n_final = sum(1 for r in results if r.get("final_answer_in_state"))
        n_answer_ready = sum(1 for r in results if r.get("stop_condition") == "answer_ready")
        print(f"  rollouts:                 {n_total}", flush=True)
        print(f"  errored:                  {n_err}", flush=True)
        print(f"  submit_answer called:     {n_submit}", flush=True)
        print(f"  final_answer in state:    {n_final}", flush=True)
        print(f"  stopped via answer_ready: {n_answer_ready}", flush=True)

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
                    f"  R{r['idx']:>2}: rollout_id={r['rollout_id']!r} "
                    f"stop={r['stop_condition']!r} "
                    f"submit={r['_curie_submit_answer_calls']} "
                    f"final={r['final_answer_in_state']} "
                    f"turns={r['root_llm_turns']}",
                    flush=True,
                )

        print("\n--- VERDICT ---", flush=True)
        if n_submit > 0 and n_final < n_submit:
            print("BUG REPRODUCED across the env_worker ZMQ subprocess boundary:", flush=True)
            print(f"  {n_submit} rollouts called submit_answer; only {n_final} ended with final_answer in state.", flush=True)
            print("  → Root cause site: the env_worker / ZMQ pipeline.", flush=True)
            print(f"  → Inspect {LOG_DIR}/env_worker_0.log + this script's stderr for", flush=True)
            print("    [CURIE-DEBUG] id_state divergence per rollout_id.", flush=True)
            verdict_code = 1
        elif n_submit == 0:
            print("INCONCLUSIVE: no rollout called submit_answer.", flush=True)
            print("  → tool_choice='required' is set; check vLLM honored it and", flush=True)
            print("    that prompts produced enough turns for the model to act.", flush=True)
            verdict_code = 2
        elif n_final == n_submit:
            print(f"HEALTHY at env_worker ZMQ subprocess boundary: {n_final}/{n_submit} terminated correctly.", flush=True)
            print("  → The verifiers env_worker subprocess is NOT the cause of Issue A.", flush=True)
            print("  → Bug requires prime-rl orchestrator's specific batching:", flush=True)
            print("    run_group, online_difficulty_filtering, max_async_level=2,", flush=True)
            print("    DAPO buffer, or trainer-side state filtering.", flush=True)
            verdict_code = 0
        else:
            print(f"PARTIAL: {n_final}/{n_submit} terminated correctly.", flush=True)
            print("  → Intermittent under ZMQ — re-run with higher REPRO_NUM_ROLLOUTS.", flush=True)
            verdict_code = 3

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
