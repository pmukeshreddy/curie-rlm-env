#!/usr/bin/env bash
# Stage 5 Continual Phase 1 — free-form current tasks.
# Quote from configs/curie_grpo_continual_phase1.toml:
# "args = { phase = 1, split = \"train\", seed = 42 }"
set -euo pipefail

if [[ -z "${INFERENCE_SERVER_IP:-}" ]]; then
    echo "ERROR: INFERENCE_SERVER_IP env var required" >&2
    exit 1
fi

# Stage 5a §11 — file descriptor limit for prime-rl API timeouts
ulimit -n 32000

uv run rl @ configs/curie_grpo_continual_phase1.toml \
    --wandb.name "${WANDB_NAME:-continual_phase1_freeform_qwen3_5_7b}" \
    --output-dir "${OUTPUT_DIR:-outputs/continual_phase1}" \
    "$@"
