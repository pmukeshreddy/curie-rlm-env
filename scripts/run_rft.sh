#!/usr/bin/env bash
# Stage 5.5 — RFT contingency launcher.
# Run only if Stage 5 phase1 stalls. See docs/STAGE_5_5_CONTINGENCY.md.
set -euo pipefail

if [[ -z "${PHASE1_STALLED_CKPT:-}" ]]; then
    echo "ERROR: PHASE1_STALLED_CKPT env var required (path to stalled phase1 ckpt)" >&2
    exit 1
fi
if [[ -z "${ROLLOUTS_DIR:-}" ]]; then
    echo "ERROR: ROLLOUTS_DIR env var required (where stalled phase1 wrote rollouts)" >&2
    exit 1
fi

ulimit -n 32000
mkdir -p outputs/rft

uv run python scripts/extract_high_reward_rollouts.py \
    --rollouts-dir "$ROLLOUTS_DIR" \
    --output outputs/rft/high_reward_rollouts.jsonl \
    --threshold "${RFT_THRESHOLD:-0.5}" \
    --max-per-task 50

uv run sft @ configs/curie_rft_phase1.toml \
    --model.name "$PHASE1_STALLED_CKPT" \
    --output-dir outputs/rft \
    --wandb.name "${WANDB_NAME:-rft_bootstrap}"

cat <<'MSG'

RFT complete. To resume Stage 5 phase1 from the bootstrapped checkpoint:
    export PHASE1_RESUME_CKPT=outputs/rft/step_<N>/
    # then re-launch run_phase1.sh with --model.name override
MSG
