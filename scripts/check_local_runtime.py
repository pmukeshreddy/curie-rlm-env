"""Diagnostic — verify CurieRLMEnv's local sandbox + inference routing.

Reports:
  * Docker daemon availability
  * Resolved sandbox backend (local_docker vs prime)
  * Local interception URL settings (sandbox→env-worker callback)
  * Presence (NOT values) of every routing-relevant env var
  * Whether `from curie_rlm_env.env import CurieRLMEnv` succeeds

Exits 0 when local mode is healthy (Docker reachable AND backend=local_docker
AND interception is local), 1 otherwise.

Usage:
    PYTHONPATH=/workspace/curie-rlm-env/src \\
        uv run --project /workspace/prime-rl python scripts/check_local_runtime.py
"""
from __future__ import annotations

import os
import sys


_ROUTING_VARS = (
    "CURIE_SANDBOX_BACKEND",
    "CURIE_USE_PRIME_TUNNEL",
    "CURIE_LOCAL_INTERCEPTION_URL",
    "CURIE_LOCAL_INTERCEPTION_HOST",
    "CURIE_LOCAL_INTERCEPTION_PORT",
    "CURIE_LOCAL_INTERCEPTION_BIND",
    "CURIE_SANDBOX_NETWORK",
    "INFERENCE_SERVER_IP",
    "INFERENCE_SERVER_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "PRIME_API_KEY",
    "HF_TOKEN",
    "WANDB_API_KEY",
    "GEMINI_API_KEY",
)

_NON_SECRET_VARS = {
    "CURIE_SANDBOX_BACKEND",
    "CURIE_USE_PRIME_TUNNEL",
    "CURIE_LOCAL_INTERCEPTION_URL",
    "CURIE_LOCAL_INTERCEPTION_HOST",
    "CURIE_LOCAL_INTERCEPTION_PORT",
    "CURIE_LOCAL_INTERCEPTION_BIND",
    "CURIE_SANDBOX_NETWORK",
    "INFERENCE_SERVER_IP",
    "OPENAI_BASE_URL",
}


def _print_env_table() -> None:
    print("=== Env var presence ===")
    for name in _ROUTING_VARS:
        present = name in os.environ and os.environ[name] != ""
        if present and name in _NON_SECRET_VARS:
            print(f"  {name}=present  value={os.environ[name]!r}")
        elif present:
            print(f"  {name}=present  (value redacted)")
        else:
            print(f"  {name}=absent")


def _check_docker() -> tuple[bool, str]:
    try:
        import docker  # type: ignore[import-untyped]
    except ImportError:
        return False, "docker Python SDK not installed (`uv pip install docker`)"
    try:
        client = docker.from_env()
        info = client.version()
    except Exception as exc:
        return False, f"daemon unreachable ({type(exc).__name__}: {exc})"
    return True, f"daemon OK (server version {info.get('Version', '?')})"


def main() -> int:
    _print_env_table()

    print()
    print("=== Sandbox backend ===")
    try:
        from curie_rlm_env.local_sandbox import resolve_sandbox_backend
    except Exception as exc:
        print(f"  ERROR: failed to import curie_rlm_env.local_sandbox: {exc}", file=sys.stderr)
        return 1
    try:
        backend = resolve_sandbox_backend()
    except ValueError as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"  CURIE_SANDBOX_BACKEND={backend}")

    print()
    print("=== Docker availability ===")
    docker_ok, docker_msg = _check_docker()
    print(f"  {'OK' if docker_ok else 'UNAVAILABLE'}: {docker_msg}")

    print()
    print("=== Local interception routing ===")
    try:
        from curie_rlm_env.env import resolve_local_interception_settings
    except Exception as exc:
        print(f"  ERROR: failed to import curie_rlm_env.env: {exc}", file=sys.stderr)
        return 1
    settings = resolve_local_interception_settings()
    if settings is None:
        print("  mode=PRIME_TUNNEL  (CURIE_USE_PRIME_TUNNEL is set)")
        interception_local = False
    else:
        print("  mode=LOCAL  (no prime_tunnel; PRIME_API_KEY NOT required)")
        print(f"  override_url={settings['override_url']!r}")
        print(f"  host={settings['host']}")
        port_note = "  (auto-assigned at first rollout)" if settings["auto_port"] else ""
        print(f"  port={settings['port']}{port_note}")
        print(f"  bind={settings['bind']}")
        interception_local = True

    print()
    print("=== CurieRLMEnv import check ===")
    try:
        from curie_rlm_env.env import CurieRLMEnv  # noqa: F401
        print("  curie_rlm_env.env.CurieRLMEnv import OK")
        env_import_ok = True
    except Exception as exc:
        print(f"  IMPORT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        env_import_ok = False

    print()
    print("=== Summary ===")
    healthy_local = (
        backend == "local_docker"
        and docker_ok
        and interception_local
        and env_import_ok
    )
    if healthy_local:
        print("  LOCAL MODE HEALTHY: PRIME_API_KEY is NOT required for training.")
        print("  Sandboxes will run on the local Docker daemon; sub-LLM callbacks")
        print("  reach the env worker via the local interception URL.")
        return 0
    print("  LOCAL MODE NOT FULLY HEALTHY:")
    if backend != "local_docker":
        print("    - sandbox backend is not 'local_docker' (PRIME_API_KEY required)")
    if not docker_ok:
        print(f"    - Docker: {docker_msg}")
    if not interception_local:
        print("    - interception is in PRIME_TUNNEL mode (PRIME_API_KEY required)")
    if not env_import_ok:
        print("    - CurieRLMEnv import failed (see above)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
