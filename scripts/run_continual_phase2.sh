#!/usr/bin/env bash
# Stage 5 Continual Phase 2 — retrieval current tasks with Phase 1 replay.
# Quote from configs/curie_grpo_continual_phase2.toml:
# "Phase 2: 70% retrieval current tasks + 30% Phase 1 replay."
set -euo pipefail

if [[ -z "${INFERENCE_SERVER_IP:-}" ]]; then
    echo "ERROR: INFERENCE_SERVER_IP env var required" >&2
    exit 1
fi
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "ERROR: GEMINI_API_KEY env var required (continual Phase 2 uses retrieval LLMSim)" >&2
    exit 1
fi
if [[ -z "${CONTINUAL_PHASE1_CKPT:-}" ]]; then
    echo "ERROR: CONTINUAL_PHASE1_CKPT env var required — point to continual Phase 1 final checkpoint path" >&2
    exit 1
fi

export CURIE_JUDGE_CACHE=1
ulimit -n 32000

uv run rl @ configs/curie_grpo_continual_phase2.toml \
    --model.name "$CONTINUAL_PHASE1_CKPT" \
    --wandb.name "${WANDB_NAME:-continual_phase2_retrieval_replay_qwen3_5_7b}" \
    --output-dir "${OUTPUT_DIR:-outputs/continual_phase2}" \
    "$@"
