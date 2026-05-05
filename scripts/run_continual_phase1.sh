#!/usr/bin/env bash
# Stage 5 Continual Phase 1 — free-form current tasks.
# Quote from configs/curie_grpo_continual_phase1.toml:
# "args = { continual_phase = 1, split = \"train\", seed = 42 }"
#
# Local-only training: PRIME_API_KEY is NOT required. CurieRLMEnv resolves a local
# sandbox→env-worker callback URL at startup (default http://127.0.0.1:<auto_port>),
# bypassing prime_tunnel. Sandboxes default to LOCAL DOCKER on the same pod
# (CURIE_SANDBOX_BACKEND=local_docker), so prime_sandboxes' hosted backend is also
# bypassed. Override via CURIE_LOCAL_INTERCEPTION_HOST / _PORT / _BIND / _URL.
# Set CURIE_USE_PRIME_TUNNEL=1 to opt into the hosted tunnel path, or
# CURIE_SANDBOX_BACKEND=prime to opt into hosted sandboxes (both require PRIME_API_KEY).
set -euo pipefail

if [[ -z "${INFERENCE_SERVER_IP:-}" ]]; then
    echo "ERROR: INFERENCE_SERVER_IP env var required (local prime-rl inference server address)" >&2
    exit 1
fi

# Stage 5a §11 — file descriptor limit for prime-rl API timeouts
ulimit -n 32000

# Local interception defaults — single-pod mode.
# Override CURIE_LOCAL_INTERCEPTION_BIND=0.0.0.0 if the sandbox runs in a separate
# network namespace (e.g. docker bridge) and needs to reach the env worker via gateway.
export CURIE_LOCAL_INTERCEPTION_HOST="${CURIE_LOCAL_INTERCEPTION_HOST:-127.0.0.1}"
export CURIE_LOCAL_INTERCEPTION_BIND="${CURIE_LOCAL_INTERCEPTION_BIND:-127.0.0.1}"
export CURIE_SANDBOX_BACKEND="${CURIE_SANDBOX_BACKEND:-local_docker}"

uv run rl @ configs/curie_grpo_continual_phase1.toml \
    --wandb.name "${WANDB_NAME:-continual_phase1_freeform_qwen3_5_7b}" \
    --output-dir "${OUTPUT_DIR:-outputs/continual_phase1}" \
    "$@"
