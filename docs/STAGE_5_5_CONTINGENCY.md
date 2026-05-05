# Stage 5.5 — RFT Contingency Runbook

One-page operator runbook. **Run only if a trigger condition fires.**

## When to run Stage 5.5

Trigger if EITHER:

1. **No reward improvement**: Continual Phase 1 (free-form current tasks) reward curve has not improved beyond `baseline_mean + 0.05` over 500 training steps in W&B.
2. **High variance**: Phase 1 step time is acceptable but reward variance is too high to yield clean gradient signal — `variance > 2x reward mean` across rollouts.

If neither trigger fires, Stage 5.5 is **not run**. The code is shelved.

## What it does

RFT (Rejection-sampling Fine-Tuning) bootstraps the policy via supervised fine-tuning on its own high-reward rollouts captured during Stage 5 continual Phase 1.

- Filters continual Phase 1's emitted rollouts by `reward >= T` (default `T = 0.5`).
- Caps each task at `max_per_task = 50` (top-K by reward).
- Writes them in prime-rl SFT format (`messages`, `task_id`, `reward` per line).
- Runs prime-rl `sft` trainer for `max_steps = 50` from the stalled continual Phase 1 checkpoint.
- Writes a new checkpoint that becomes the resume point for continual Phase 1.

Every demonstration is a real Qwen self-rollout on a real Curie problem. **ZERO-SYNTHETIC compliant** per CLAUDE.md L15.

## How to run

```bash
# Required env vars
export PHASE1_STALLED_CKPT=outputs/continual_phase1/step_<N>/      # the stalled checkpoint
export ROLLOUTS_DIR=outputs/continual_phase1/rollouts/             # continual Phase 1 rollout records

# Optional
export RFT_THRESHOLD=0.5                                  # default; lower if filtered set is empty
export WANDB_NAME=rft_bootstrap                           # default

./scripts/run_rft.sh
```

The script:
1. Runs `extract_high_reward_rollouts.py` to filter + cap → `outputs/rft/high_reward_rollouts.jsonl`.
2. Runs `uv run sft @ configs/curie_rft_phase1.toml` with `--model.name $PHASE1_STALLED_CKPT`.
3. Prints resume instructions for continual Phase 1.

## How to verify it worked

After RFT completes:

1. Resume continual Phase 1 with the RFT'd checkpoint:
   ```bash
   export PHASE1_RESUME_CKPT=outputs/rft/step_<N>/
   OUTPUT_DIR=outputs/continual_phase1_rft_resume \
       ./scripts/run_continual_phase1.sh --model.name "$PHASE1_RESUME_CKPT"
   ```
2. Within the first ~50 steps of the resumed run, verify in W&B:
   - **Structured trajectories**: assistant turns include tool calls (not pure text).
   - **Sub-LM call count > 0**: `llm_batch` is being invoked.
   - **Reward variance reduced**: per-step variance ÷ mean is lower than pre-RFT.

If all three hold, RFT helped. Continue continual Phase 1 → continual Phase 2 → continual Phase 3.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: RFT requires at least N high-reward rollouts; got 0` | Threshold too high, or phase 1 hasn't produced any successful rollouts. | Run more phase 1 steps OR lower `RFT_THRESHOLD=0.3`. |
| SFT loss explodes (NaN / >10) within first 5 steps | `lr=5e-7` too high for the stalled checkpoint state. | Drop to `lr=1e-7` via `--trainer.optim.lr 1e-7`. |
| Resumed phase 1 still doesn't learn | RFT didn't help — model can't make the leap from stalled state. | **Escalate**: collect rollout examples for human review, consider Stage 8 hub push and external eyes. |
| `PHASE1_STALLED_CKPT` missing or invalid | Operator forgot the env var. | Set the var; the bash launcher hard-fails with a clear message. |
| `extract_high_reward_rollouts.py` skips most records | prime-rl rollout schema differs from our schema-agnostic normalizer. | Inspect a few `*.jsonl` records under `$ROLLOUTS_DIR`; adjust `_normalize_record()` in the script. |

## Provenance

- Trigger conditions, RFT format, and ZERO-SYNTHETIC reasoning are locked in `CLAUDE.md` "Documented Deviations" section.
- Hyperparameter choices (`lr=5e-7`, `loss_mask=assistant_only`, `max_steps=50`) source from Stage 4a memo §3-4 + Stage 5b precedent (lower than phase 1's `lr=1e-6`).
- Schema-agnostic extractor is a Stage 5b OQ-A precedent (prime-rl rollout output schema not verified from web docs).
