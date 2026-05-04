#!/usr/bin/env bash
# Stage 5b Phase 2 — retrieval RL training. LLMSim in loop. Judge cache active.
# Resumes from Phase 1 checkpoint (set PHASE1_CKPT env var).
set -euo pipefail

if [[ -z "${INFERENCE_SERVER_IP:-}" ]]; then
    echo "ERROR: INFERENCE_SERVER_IP env var required" >&2
    exit 1
fi
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "ERROR: GEMINI_API_KEY env var required (Phase 2 uses LLMSim with Gemini-2.5-Pro)" >&2
    exit 1
fi
if [[ -z "${PHASE1_CKPT:-}" ]]; then
    echo "ERROR: PHASE1_CKPT env var required — point to Phase 1 final checkpoint path" >&2
    exit 1
fi

# Signal cache semantics to operators (cached_llmsim_sync is always-on; the var
# is documentary — Stage 5b judge_cache.py + scorers.py llm_sim() integration).
export CURIE_JUDGE_CACHE=1
ulimit -n 32000

uv run rl @ configs/curie_grpo_retrieval.toml \
    --model.name "$PHASE1_CKPT" \
    --wandb.name "${WANDB_NAME:-phase2_retrieval_qwen3_5_7b}" \
    --output-dir "${OUTPUT_DIR:-outputs/phase2}"
