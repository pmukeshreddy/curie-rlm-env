#!/usr/bin/env bash
# Stage 5b Phase 3 — geometric/structural RL training. Programmatic rewards only.
# Resumes from Phase 2 checkpoint (set PHASE2_CKPT env var).
set -euo pipefail

if [[ -z "${INFERENCE_SERVER_IP:-}" ]]; then
    echo "ERROR: INFERENCE_SERVER_IP env var required" >&2
    exit 1
fi
if [[ -z "${PHASE2_CKPT:-}" ]]; then
    echo "ERROR: PHASE2_CKPT env var required — point to Phase 2 final checkpoint path" >&2
    exit 1
fi

ulimit -n 32000

uv run rl @ configs/curie_grpo_geometric.toml \
    --model.name "$PHASE2_CKPT" \
    --wandb.name "${WANDB_NAME:-phase3_geometric_qwen3_5_7b}" \
    --output-dir "${OUTPUT_DIR:-outputs/phase3}"
