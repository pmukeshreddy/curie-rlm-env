# AGENTS.md — curie-rlm-env (Locked Rules — 2026 Best Practices)

## Project Overview
Private research project implementing the official Google CURIE scientific long-context benchmark (10 tasks, 580 problems) using Recursive Language Models (RLM) + GRPO.  
Policy model: Qwen3.5-7B. All recursion/long-context/code execution inside Prime Sandboxes + official RLMEnv.

## Mandatory Stack (Never deviate)
- verifiers (RLMEnv from verifiers.envs.experimental.rlm_env; RubricGroup + JudgeRubric at top level; sandbox via SandboxMixin; monitor rubrics auto-attached by RLMEnv)
- prime-rl (cloned + installed separately on GPU pod at Stage 5 per official docs; not a pyproject dependency)
- Python ~= 3.12.0
- Qwen3.5-7B as policy model

## ZERO-FALLBACK + ANTI-HALLUCINATION + ANTI-SYNTHETIC + ANTI-CHEATING RULES (VIOLATION = INSTANT REJECT)
- NEVER add fallbacks, try/catch, defaults, silent errors, or "just in case" logic.
- NEVER create synthetic data, fake trajectories, or made-up examples.
- Every claim or code change MUST start with word-for-word quotes from actual files or official Prime/Curie documentation.
- If uncertain → reply "I don't have enough information" and ask. Never guess.
- Always read target file(s) first before reasoning.

## Curie Metrics & Rubric (Verbatim from Paper + curie_run_eval.ipynb)
Rubric must act as dispatcher:
- Retrieval (DFT-S, DFT-P, MPVE) → LLMSim (precision/recall/F1)
- Geometric (BIOGR) → IoU
- Structural (PDB) → Identity ratio (ID_r) — FASTA `>` path only; code-exec dropped
- Free-form (DFT-C, HFE, HFD, QECC_65, GEO) → ROUGE-L + BERTScore

## Documented Deviations from Curie release
- PDB scorer: `pdb_execute_code_eval` `exec()` branch dropped for sandbox safety. FASTA `>` extraction path only.
- Free-form: BERTScore (Curie's released `_SHARED_METRCS`) replaces paper-only LMScore (no implementation in `colabs/curie_run_eval.ipynb`).
- DFT-C: scored as free-form (ROUGE-L + BERTScore) per Curie cell 30 — `_FULL_ADDITIONAL_METRICS["dft"]` registers LLMSim only for DFT-S/P, not DFT-C.
- Stage 4 (SFT) skipped — no SFT data for RLM on Curie. Cold-start GRPO per DeepSeek-R1-Zero precedent. Stage 5.5 RFT is contingency.

Algorithm naming:
- prime-rl default loss is DPPO+KL with DR-GRPO advantages (per docs/bring-your-own-algorithms.md, Stage 5a memo).
- Hyperparameters use prime-rl names: kl_tau, dppo_mask_high, dppo_mask_low, adv_tau (NOT classical kl_coef, clip_range, beta).
- "GRPO" used colloquially for the family. Precise name when reporting: DPPO+KL on prime-rl with DR-GRPO advantages.

Stage 5 RL training is continual replay, not sequential family-only training:
- Phase 1 — 100% current free-form tasks: DFT-C, HFE, HFD, QECC_65, GEO
- Phase 2 — 70% current retrieval tasks (DFT-S, DFT-P, MPVE) + 30% Phase 1 replay
- Phase 3 — 60% current geometric/structural tasks (BIOGR, PDB) + 20% Phase 1 replay + 20% Phase 2 replay
Each phase resumes from the prior phase's checkpoint, and replay is part of the default training data path.

Stage 5.5 RFT contingency:
- Triggered only if Stage 5 continual Phase 1 stalls (no reward improvement over baseline + 0.05 in 500 steps, OR variance > 2x reward mean).
- RFT uses the stalled policy's OWN high-reward rollouts (reward >= threshold T, default 0.5) as SFT data.
- This is rejection-sampling fine-tuning. Every demonstration is a real Qwen execution on a real Curie problem. ZERO-SYNTHETIC compliant.
- After RFT, Stage 5 continual Phase 1 resumes from the RFT'd checkpoint.

Stage 6 skipped. Original placeholder ("mixing + online refinement") had no concrete deliverable. Continual replay now lives inside Stage 5 itself; prime-rl handles async off-policy correction internally. Final pipeline: 0 → 1 → 2 → 3 → 3.5 → [4 skip] → 5 continual replay → 5.5 (contingency) → [6 skip] → 6.5 (verification) → 7 → 8 (optional).

Stage 7 = internal results only. External comparisons (Curie paper baselines, frontier models) deferred to post-Stage-7 narrative work once internal numbers are in hand. Report generator hard-fails on missing required eval JSON keys (no soft 0.0 defaults). std is optional (rendered "n/a" if absent on small-N tasks).

BERTScore baseline calibration:
- Curie uses bert_score(lang="en") verbatim — we inherit this.
- Random English vs scientific text scores ~0.7-0.85 (uncalibrated baseline).
- Garbage-vs-content gradient on free-form tasks is therefore small (floor ~0.4-0.5 with our 0.5+0.5 ROUGE/BERT weighting).
- Stage 5 watch item: if free-form reward curves flatten during GRPO, evaluate bert_score(rescale_with_baseline=True) as a calibration fix.
- Not blocking; documented for future debugging continuity.

## Stage 3 reward design (locked)
Zero anti-hack reward functions in CurieRubric. Curie's formulas (ROUGE/BERT/IoU/ID_r) are length-bounded structurally; observability via auto-attached `RLMMonitorRubric` plus weight-0 ROUGE+BERT auxiliary metrics on all 10 tasks. Defenses are added with W&B evidence in Stage 5+, never preemptively.

Headline metric: per-task normalized score → average across 10 tasks (matches Figure 5).  
Optional: pass@k (threshold-based, e.g. F1>0.5 for retrieval, ROUGE-L>0.3 for free-form).

## Reward Hacking Guards (Strictly Enforced)
1. Freeze judge model (Gemini or Codex — different family from Qwen policy).
2. Programmatic spot-check after LLMSim (numeric tolerance <5%).
3. Length penalty for outputs significantly longer than ground-truth average (~954 words).
4. Use exact frozen prompts from Curie repo (never "improve").
5. Sanity batch every ~100 GRPO steps (log 5 random rollouts).
6. Reward weighting in Rubric:
   - Programmatic (IoU, ID_r, ROUGE-L) → weight 1.0
   - LLMSim → weight 0.7
   - BERTScore → weight 0.5

## Sub-Agent Policy (Automatic)
For any task with 2+ independent parts, act as orchestrator and spawn minimum single-responsibility sub-agents in parallel: brainstormer, reviewer, tester.

## Stage 0 Safeguards (Enforced Everywhere)
- sub_llm_max_turns: 1
- per-step token budget + sandbox limits (PrimeSandbox)
- Model must always return {"ready": True} in answer dict
- Strict train/val/test splits — zero leakage

See @README.md for full pipeline and metrics.
See config/safeguards.yaml for exact locked values.
