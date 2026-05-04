#!/usr/bin/env bash
# Stage 5b Phase 1 — free-form RL training.
# Programmatic rewards only (ROUGE-L + BERTScore). No Gemini calls.
# Resumes from base Qwen3.5-7B (no prior checkpoint).
set -euo pipefail

if [[ -z "${INFERENCE_SERVER_IP:-}" ]]; then
    echo "ERROR: INFERENCE_SERVER_IP env var required" >&2
    exit 1
fi

# Stage 5a §11 — file descriptor limit for prime-rl API timeouts
ulimit -n 32000

# Phase 1 starts from base policy. Override --model.name if a custom mirror is needed.
uv run rl @ configs/curie_grpo_freeform.toml \
    --wandb.name "${WANDB_NAME:-phase1_freeform_qwen3_5_7b}" \
    --output-dir "${OUTPUT_DIR:-outputs/phase1}"
