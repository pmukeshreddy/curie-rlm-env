#!/usr/bin/env bash
# Stage 5 Continual Phase 3 — geometric/structural current tasks with Phase 1+2 replay.
# Quote from configs/curie_grpo_continual_phase3.toml:
# "Phase 2 replay: DFT-S, DFT-P, MPVE."
#
# Local-only training: PRIME_API_KEY is NOT required (see run_continual_phase1.sh
# header for the local-interception env-var contract).
set -euo pipefail

if [[ -z "${INFERENCE_SERVER_IP:-}" ]]; then
    echo "ERROR: INFERENCE_SERVER_IP env var required (local prime-rl inference server address)" >&2
    exit 1
fi
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "ERROR: GEMINI_API_KEY env var required (continual Phase 3 replays retrieval LLMSim tasks)" >&2
    exit 1
fi
if [[ -z "${CONTINUAL_PHASE2_CKPT:-}" ]]; then
    echo "ERROR: CONTINUAL_PHASE2_CKPT env var required — point to continual Phase 2 final checkpoint path" >&2
    exit 1
fi

export CURIE_JUDGE_CACHE=1
export CURIE_LOCAL_INTERCEPTION_HOST="${CURIE_LOCAL_INTERCEPTION_HOST:-127.0.0.1}"
export CURIE_LOCAL_INTERCEPTION_BIND="${CURIE_LOCAL_INTERCEPTION_BIND:-127.0.0.1}"
ulimit -n 32000

uv run rl @ configs/curie_grpo_continual_phase3.toml \
    --model.name "$CONTINUAL_PHASE2_CKPT" \
    --wandb.name "${WANDB_NAME:-continual_phase3_geometric_structural_replay_qwen3_5_7b}" \
    --output-dir "${OUTPUT_DIR:-outputs/continual_phase3}" \
    "$@"
