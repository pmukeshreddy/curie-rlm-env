"""Diagnostic — verify CurieRLMEnv routes rollouts locally (no prime_tunnel).

Reports presence (NOT values) of every env var that controls local inference
routing and the override URL CurieRLMEnv would compute. Local routing is the
only supported mode; this script exits 0 when settings resolve cleanly and 1
when CurieRLMEnv cannot be imported or the settings are misconfigured.

Usage:
    PYTHONPATH=/workspace/curie-rlm-env/src \\
        uv run --project /workspace/prime-rl python scripts/check_local_inference_routing.py
"""
from __future__ import annotations

import os
import sys


_ROUTING_VARS = (
    "CURIE_LOCAL_INTERCEPTION_URL",
    "CURIE_LOCAL_INTERCEPTION_HOST",
    "CURIE_LOCAL_INTERCEPTION_PORT",
    "CURIE_LOCAL_INTERCEPTION_BIND",
    "INFERENCE_SERVER_IP",
    "INFERENCE_SERVER_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "HF_TOKEN",
    "WANDB_API_KEY",
    "GEMINI_API_KEY",
)

_NON_SECRET_VARS = {
    "CURIE_LOCAL_INTERCEPTION_URL",
    "CURIE_LOCAL_INTERCEPTION_HOST",
    "CURIE_LOCAL_INTERCEPTION_PORT",
    "CURIE_LOCAL_INTERCEPTION_BIND",
    "INFERENCE_SERVER_IP",
    "OPENAI_BASE_URL",
}


def main() -> int:
    print("=== Local inference routing — env var presence ===")
    for name in _ROUTING_VARS:
        present = name in os.environ and os.environ[name] != ""
        if present and name in _NON_SECRET_VARS:
            print(f"  {name}=present  value={os.environ[name]!r}")
        elif present:
            print(f"  {name}=present  (value redacted)")
        else:
            print(f"  {name}=absent")

    print()
    print("=== Resolved CurieRLMEnv local-routing settings ===")
    try:
        from curie_rlm_env.env import resolve_local_interception_settings
    except Exception as e:
        print(f"  ERROR: failed to import curie_rlm_env.env: {e}", file=sys.stderr)
        return 1

    try:
        settings = resolve_local_interception_settings()
    except ValueError as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"  override_url={settings['override_url']!r}")
    print(f"  host={settings['host']}")
    print(f"  port={settings['port']}{'  (auto-assigned at first rollout)' if settings['auto_port'] else ''}")
    print(f"  bind={settings['bind']}")

    print()
    print("Importing CurieRLMEnv class symbol...")
    try:
        from curie_rlm_env.env import CurieRLMEnv  # noqa: F401
    except Exception as exc:
        print(f"  IMPORT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("  curie_rlm_env.env.CurieRLMEnv import OK")

    print()
    print("Routing summary: local-only mode (the only supported mode).")
    print("  Sandbox sub-LLM callbacks → env worker interception server at")
    if settings["auto_port"]:
        print(f"    http://{settings['host']}:<auto_port>  (port set after server bind)")
    else:
        print(f"    {settings['override_url']}")
    print("  Policy completions → local prime-rl inference server (INFERENCE_SERVER_IP).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
