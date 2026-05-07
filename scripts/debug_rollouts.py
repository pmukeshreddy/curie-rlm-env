"""Rollouts ONLY — prime-rl's exact plumbing, no trainer, no DPPO+KL.

This pod's prime-rl is too old for `prime_rl.eval`, so we mirror what
prime_rl/orchestrator/env_worker.py:process_request does (the function
that runs inside the orchestrator's env-worker subprocess during real
training) — minus the IPC/transport/trainer hookup.

Same `setup_clients` from prime_rl.utils.client → identical AsyncOpenAI
wiring to Stage 5 training. Same `vf.load_environment("curie-rlm-env",
continual_phase=N)` → identical CurieRLMEnv. Same `env.run_group(...)`
call shape. So any [CURIE-DEBUG] line that fires here is what training
sees.

Usage (from /workspace/curie-rlm-env on the pod):

    PYTHONPATH=/workspace/curie-rlm-env/src \\
      uv run --project /workspace/prime-rl \\
      python scripts/debug_rollouts.py \\
      --base-url http://localhost:8000/v1

Then:

    grep '\\[CURIE-DEBUG\\]' results/debug_rollouts/*.stderr.log | head -40
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import httpx
import verifiers as vf
from openai import AsyncOpenAI


def _make_async_openai_like_prime_rl(
    base_url: str,
    api_key_var: str,
    timeout_seconds: int = 1200,
) -> AsyncOpenAI:
    """Construct AsyncOpenAI exactly as prime_rl.utils.client.setup_clients does.

    Inlined (not imported) because the pod's prime-rl version splits configs
    into a sub-package (prime-rl-configs) and the import path for ClientConfig
    differs across releases. The body below mirrors setup_clients verbatim:
    same httpx limits, same timeout shape, same max_retries, same EMPTY
    fallback for the api key. If the orchestrator can hit the vLLM server,
    so can this client.
    """
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=8192, max_keepalive_connections=8192),
        timeout=httpx.Timeout(timeout_seconds),
    )
    return AsyncOpenAI(
        base_url=base_url,
        api_key=os.getenv(api_key_var, "EMPTY"),
        max_retries=10,
        http_client=http_client,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rollouts-only debug probe through prime-rl's setup_clients."
    )
    p.add_argument("--continual-phase", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-examples", type=int, default=2,
                   help="How many distinct dataset examples to roll out (default 2)")
    p.add_argument("--rollouts-per-example", type=int, default=1)
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="vLLM OpenAI-compatible base URL (must already be serving the model)")
    p.add_argument("--api-key-var", default="OPENAI_API_KEY",
                   help="Env var holding the API key (default OPENAI_API_KEY; falls back to 'EMPTY')")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--out-dir", default="results/debug_rollouts")
    return p.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # AsyncOpenAI built the same way prime_rl.utils.client.setup_clients does
    # (inlined — see _make_async_openai_like_prime_rl docstring for why).
    async_openai = _make_async_openai_like_prime_rl(args.base_url, args.api_key_var)
    # The pod's verifiers env.run_group calls resolve_client(client), which on
    # this version rejects raw AsyncOpenAI ("Unsupported client type"). Wrap
    # it in vf.OpenAIChatCompletionsClient so resolve_client accepts it.
    client = vf.OpenAIChatCompletionsClient(async_openai)
    print(f"[debug] vf.OpenAIChatCompletionsClient wrapping AsyncOpenAI "
          f"base_url={args.base_url} model={args.model}")

    env = vf.load_environment(
        "curie-rlm-env",
        continual_phase=args.continual_phase,
        split=args.split,
        seed=args.seed,
    )
    print(f"[debug] env loaded continual_phase={args.continual_phase} "
          f"split={args.split} dataset_size={len(env.dataset)}")

    sampling_args: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    n_examples = min(args.num_examples, len(env.dataset))
    rows: list[dict[str, Any]] = []

    for ex_idx in range(n_examples):
        example = dict(env.dataset[ex_idx])  # plain dict copy
        log_path = out_dir / f"example_{ex_idx}.stderr.log"
        print(f"[debug] running example {ex_idx + 1}/{n_examples} → {log_path}", flush=True)

        with open(log_path, "w") as log_fh:
            with contextlib.redirect_stderr(log_fh):
                try:
                    # Signature on the pod's verifiers (confirmed via TypeError):
                    #   run_group(group_inputs, client, model, sampling_args, ...)
                    # `sampling_args` is positional/required; older `gen_sampling_args`
                    # / `gen_sem` / `score_sem` kwargs are NOT in this version.
                    states = await env.run_group(
                        group_inputs=[
                            vf.RolloutInput(**example)
                            for _ in range(args.rollouts_per_example)
                        ],
                        client=client,
                        model=args.model,
                        sampling_args=sampling_args,
                    )
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc(file=sys.stderr)
                    rows.append({
                        "example": ex_idx,
                        "rollout": "-",
                        "has_final_answer": "ERR",
                        "submit_calls": "-",
                        "n_turns": "-",
                        "reward": "-",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    continue

        for r_idx, state in enumerate(states):
            rows.append({
                "example": ex_idx,
                "rollout": r_idx,
                "has_final_answer": "final_answer" in state,
                "submit_calls": state.get("_curie_submit_answer_calls", 0),
                "n_turns": len(state.get("trajectory") or []),
                "reward": state.get("reward"),
                "error": state.get("error"),
                "trajectory": state.get("trajectory") or [],
                "final_answer": state.get("final_answer"),
            })


def _dump_trajectory(r: dict[str, Any]) -> None:
    """Print the full per-turn record: prompt, assistant message, tool calls, tool responses."""
    traj = r.get("trajectory") or []
    print(f"\n--- TRAJECTORY example={r['example']} rollout={r['rollout']} "
          f"({len(traj)} turns) ---")
    for t_idx, step in enumerate(traj):
        prompt = step.get("prompt") if isinstance(step, dict) else None
        completion = step.get("completion") if isinstance(step, dict) else None
        print(f"\n  [turn {t_idx}]")
        if prompt:
            # prompt is the messages going INTO the model on this turn
            for m_idx, m in enumerate(prompt if isinstance(prompt, list) else [prompt]):
                role = (getattr(m, "role", None)
                        or (m.get("role") if isinstance(m, dict) else None))
                content = (getattr(m, "content", None)
                           or (m.get("content") if isinstance(m, dict) else None))
                content_repr = str(content)[:400] if content else ""
                print(f"    PROMPT[{m_idx}] role={role!r} content={content_repr!r}")
        if completion:
            # completion is what the model produced on this turn (assistant + tool messages)
            comp_list = completion if isinstance(completion, list) else [completion]
            for m_idx, m in enumerate(comp_list):
                role = (getattr(m, "role", None)
                        or (m.get("role") if isinstance(m, dict) else None))
                content = (getattr(m, "content", None)
                           or (m.get("content") if isinstance(m, dict) else None))
                tool_calls = (getattr(m, "tool_calls", None)
                              or (m.get("tool_calls") if isinstance(m, dict) else None))
                content_repr = str(content)[:600] if content else ""
                print(f"    COMPL[{m_idx}] role={role!r} content={content_repr!r}")
                if tool_calls:
                    for tc_idx, tc in enumerate(tool_calls):
                        tc_name = (getattr(tc, "name", None)
                                   or (tc.get("name") if isinstance(tc, dict) else None))
                        tc_args = (getattr(tc, "arguments", None)
                                   or (tc.get("arguments") if isinstance(tc, dict) else None))
                        print(f"      tool_call[{tc_idx}] name={tc_name!r} "
                              f"arguments={str(tc_args)[:400]!r}")
                else:
                    print(f"      tool_calls=NONE")
    print(f"\n  final_answer={r.get('final_answer')!r}")

    # Summary table — short columns first; full error dumps below.
    print()
    print("=" * 96)
    print(f"{'ex':>3} {'r':>3} {'has_final':>10} {'submit':>7} {'turns':>5} {'reward':>8}  error_short")
    print("-" * 96)
    for r in rows:
        rew = f"{r['reward']:.3f}" if isinstance(r["reward"], (int, float)) else str(r["reward"])
        err_short = (str(r["error"])[:40] + "…") if r["error"] and len(str(r["error"])) > 40 else (r["error"] or "")
        print(
            f"{str(r['example']):>3} {str(r['rollout']):>3} {str(r['has_final_answer']):>10} "
            f"{str(r['submit_calls']):>7} {str(r['n_turns']):>5} {rew:>8}  {err_short}"
        )
    print("=" * 96)

    # Full per-row error dump — the truncated `error_short` above is useless
    # when the error is a chained ModelError with HTTP status + server body.
    print("\nFULL ERROR DUMPS:")
    for r in rows:
        if not r["error"]:
            continue
        print(f"  example={r['example']} rollout={r['rollout']}")
        if isinstance(r["error"], dict):
            for k, v in r["error"].items():
                print(f"    {k}: {v}")
        else:
            print(f"    {r['error']}")

    # Full per-turn dump: prompt, assistant content, tool calls, tool responses.
    print("\nFULL TRAJECTORIES:")
    for r in rows:
        if "trajectory" in r:
            _dump_trajectory(r)

    n_with_answer = sum(1 for r in rows if r["has_final_answer"] is True)
    n_with_submit = sum(
        1 for r in rows if isinstance(r["submit_calls"], int) and r["submit_calls"] > 0
    )
    print(f"\n{n_with_answer}/{len(rows)} rollouts produced has_final_answer=True")
    print(f"{n_with_submit}/{len(rows)} rollouts called submit_answer at least once")
    print(f"\n[CURIE-DEBUG] env+sandbox traces are in {out_dir}/example_*.stderr.log")
    print(f"  grep '\\[CURIE-DEBUG\\]' {out_dir}/example_*.stderr.log")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
