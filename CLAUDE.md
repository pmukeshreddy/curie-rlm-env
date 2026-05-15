# CLAUDE.md — curie-rlm-env (Locked Rules — 2026 Best Practices)

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
- PDB id_r length-floored + length-normalized: `scorers.id_r` adds an absolute 30-residue floor and a 0.3-fraction-of-reference floor BEFORE alignment; predictions below either floor return `identity_ratio=0.0` and `length_floor_rejected=True` (the latter is a new return-dict field, logged for Stage 5 W&B). Denominator switched from `len(best_alignment[0])` (alignment length with internal gaps) to `max(len_pred, len_ref)` for an explicit length-normalized identity. Aligner scoring is set explicitly to match=1/mismatch=0/gap=-1 (Biopython default, but pinned so a future library change can't silently grant BLOSUM credit). Strict-identity rather than biological-similarity is the right RL reward shape — note this distinction is intentional.
- BIOGR DIoU replaces plain IoU: `scorers.diou` (renamed from `iou`) implements Distance-IoU (Zheng et al. 2020): `DIoU = IoU - ρ²(centers) / c²(enclosing diagonal)`, output in [-1, 1]. Closes the sparse-gradient cliff of plain IoU (which returns exactly 0 for any non-overlap regardless of how close). Antimeridian-crossing bboxes (W>E) are split into [W,180] and [-180,E] sub-bboxes before intersection/union; centers computed in a shifted frame. Strict bbox validation: S>=N, |lat|>90, zero-area bboxes raise ValueError instead of silently scoring 0. Reward clamped to [0, 1] at `rubric._diou_reward` consumption site (same pattern as rescaled-BERT clamp); raw negative output reaches `_aux_diou_raw` for Stage 5 W&B diagnostics.
- ROUGE-L stopword filtering: `scorers.rouge_l` wires a custom Tokenizer that drops English function words (NLTK stopword list, embedded inline — no nltk-data download required at runtime) AFTER the library's stemming step. Empirically the stopword-only content-free baseline drops from ~13/100 to 0/100 on scientific GT; legitimate content-rich predictions score within a few % of the pre-filter result. Originally triggered by the interaction with the free-form geometric coupling: rescaled BERT was meant to clamp the semantic-only hack, but stopword-grift on ROUGE-L could still produce non-trivial reward. After the rescale revert (above), the stopword filter is the only remaining anti-grift signal on the ROUGE-L side and is therefore retained.
- LLMSim numeric-value verifier: `scorers.llm_sim` post-filters Gemini's per-GT match decisions with a deterministic numeric verifier (5% relative tolerance, CLAUDE.md guard #2; tolerance locked in `config/safeguards.yaml:43`). Closes the within-record verbosity-grift pathway where Gemini accepts a "2.1 eV (measured ... reported in Table 3)" string as matching the GT "2.1 eV" but would also accept "2.5 eV (same prose dressing)" — the verifier revokes the second. Cell 18 verbatim `num_match` logic is preserved upstream of the filter; only the count of matches changes. New return key: `verifier_revoked_count`. Filter, not generator: cannot add matches Gemini didn't claim.
- BERTScore for free-form: Curie cell 20 verbatim — `BERTScorer(lang="en")` with library defaults (no baseline rescale). The earlier Stage 5 deviation `rescale_with_baseline=True` was REVERTED based on Stage 3b ZMQ harness W&B evidence: 16/16 baseline Qwen3-8B rollouts on Phase 1 free-form data produced rescaled BERT_F1 in [-0.77, -0.16] (uniformly below the random English baseline). The `max(0.0, raw)` clamp in `_freeform_geometric_reward` zeroed every rollout, the geometric mean's zero-guard collapsed every reward, DAPO `online_difficulty_filtering=true` rejected every group, and the trainer was stuck at step 0 forever. CLAUDE.md L62 ("defenses are added with W&B evidence in Stage 5+, never preemptively") is the rule that mandates this revert: rescaling was a preemptive anti-length-grift defense, the empirical evidence shows it kills the reward signal, so the revert restores Curie's documented behavior. Length-grift remains a Stage 5 watch item and will be addressed with a targeted output-length cap (or rescaling re-enabled with a different consumer-side combiner) once we have W&B evidence of length-grift, not before.
- Free-form reference extraction (`CurieRubric._freeform_reference`): the dataset's `answer` field for free-form tasks is `json.dumps(entry["ground_truth"])` (datasets.py:148), and the GT is a structured object — for DFT-C a dict with `{code, graph_as_text, no_header_code, record_id}`, for HFE a dict with `{Hamiltonian, Other_info, Score, arxiv_id, record_id}`, for HFD a list of step dicts, for QECC_65 a list of `{code_id, physical, name, ...}` records, for GEO a dict with `{datasets, notes, paper_link, paper_title, record_id}`. Comparing model prose against `json.dumps(...)` of those structures passes JSON syntax (`{`, `}`, `"key":`) and identifier metadata (record_id, arxiv_id, paper_link, code_id) into ROUGE/BERT — both pure noise that compresses the score distribution. The rubric now extracts content per task before scoring: DFT-C uses the TASK_MAP-backed `code` field; the other four use a generic recursive collector that pulls every string value from the GT object while skipping identifier-like keys (anything ending in `_id`, plus `id`, `url`, `doi`, `paper_link`). Falls back to the original `answer` string when the GT can't be parsed as JSON or extraction yields nothing — explicit "extraction did not apply" semantics, not a silent failure. Applied identically in `_freeform_geometric_reward` (headline) and `_aux_rouge_lsum`/`_aux_bert_f1` (observability) so headline and aux numbers stay consistent. Discovered after the BERTScore rescale revert when the Stage 3b harness showed BERT raw clustering at 0.72-0.80 across all 16 rollouts; investigation traced part of the compression to the json-encoded reference (the rest is the structural prose-vs-code/LaTeX/dict mismatch which only changes as the policy learns to produce GT-shaped outputs).

Algorithm naming:
- prime-rl default loss is DPPO+KL with DR-GRPO advantages (per docs/bring-your-own-algorithms.md, Stage 5a memo).
- Hyperparameters use prime-rl names: kl_tau, dppo_mask_high, dppo_mask_low, adv_tau (NOT classical kl_coef, clip_range, beta).
- "GRPO" used colloquially for the family. Precise name when reporting: DPPO+KL on prime-rl with DR-GRPO advantages.

Stage 5 RL training is continual replay, not sequential family-only training:
- Phase 1 — 100% current free-form tasks: DFT-C, HFE, HFD, QECC_65, GEO
- Phase 2 — 70% current retrieval tasks (DFT-S, DFT-P, MPVE) + 30% Phase 1 replay
- Phase 3 — 60% current geometric/structural tasks (BIOGR, PDB) + 20% Phase 1 replay + 20% Phase 2 replay
Each continual phase resumes from the prior continual phase's checkpoint, and replay is part of the default training data path.

Stage 5.5 RFT contingency:
- Triggered only if Stage 5 continual Phase 1 stalls (no reward improvement over baseline + 0.05 in 500 steps, OR variance > 2x reward mean).
- RFT uses the stalled policy's OWN high-reward rollouts (reward >= threshold T, default 0.5) as SFT data.
- This is rejection-sampling fine-tuning. Every demonstration is a real Qwen execution on a real Curie problem. ZERO-SYNTHETIC compliant.
- After RFT, Stage 5 continual Phase 1 resumes from the RFT'd checkpoint.

Stage 6 skipped. Original placeholder ("mixing + online refinement") had no concrete deliverable. Continual replay now lives inside Stage 5 itself; prime-rl handles async off-policy correction internally. Final pipeline: 0 → 1 → 2 → 3 → 3.5 → [4 skip] → 5 continual replay → 5.5 (contingency) → [6 skip] → 6.5 (verification) → 7 → 8 (optional).

Stage 7 = internal results only. External comparisons (Curie paper baselines, frontier models) deferred to post-Stage-7 narrative work once internal numbers are in hand. Report generator hard-fails on missing required eval JSON keys (no soft 0.0 defaults). std is optional (rendered "n/a" if absent on small-N tasks).

BERTScore baseline calibration:
- Curie uses bert_score(lang="en") verbatim — we inherit this (raw, no rescale).
- Random English vs scientific text scores ~0.7-0.85 (uncalibrated baseline).
- Garbage-vs-content gradient on free-form tasks is therefore small.
- Stage 5 W&B evidence (Stage 3b ZMQ harness, 16/16 baseline rollouts, 2026-05-14): rescale_with_baseline=True forced ALL Phase 1 rollouts below baseline (rescaled F1 in [-0.77, -0.16]) and zeroed every reward, hanging the trainer at step 0. The rescaling deviation has been REVERTED — see the BERTScore entry under "Documented Deviations from Curie release" above.
- Length-grift watch: if a real W&B trace (Stage 5, post-revert) shows length-grift drift, address with a per-task output-length cap inside `_freeform_geometric_reward` (or revisit the consumer-side combiner so a re-enabled rescale doesn't zero the reward). Do not silently re-enable rescaling without a paired combiner change.

## Stage 3 reward design (locked)
Zero anti-hack reward functions in CurieRubric. Curie's formulas (ROUGE/BERT/IoU/ID_r) are length-bounded structurally; observability via auto-attached `RLMMonitorRubric` plus weight-0 ROUGE+BERT auxiliary metrics on all 10 tasks. Defenses are added with W&B evidence in Stage 5+, never preemptively.

Headline metric: per-task normalized score → average across 10 tasks (matches Figure 5).  
Optional: pass@k (threshold-based, e.g. F1>0.5 for retrieval, ROUGE-L>0.3 for free-form).

## Reward Hacking Guards (Strictly Enforced)
1. Freeze judge model (Gemini or Claude — different family from Qwen policy).
2. Programmatic spot-check after LLMSim (numeric tolerance <5%). Implemented
   in `scorers.llm_sim` as a deterministic numeric-value verifier that
   FILTERS Gemini's per-GT match decisions — it can revoke matches where any
   overlapping numeric field disagrees beyond tolerance, but never adds
   matches Gemini didn't claim. `verifier_revoked_count` logged per rollout
   in the return dict. In-scope unit families: energy, length, inverse-length,
   temperature (°C excluded — affine conversion out of scope).
3. Length penalty for outputs significantly longer than ground-truth average (~954 words).
4. Use exact frozen prompts from Curie repo (never "improve").
5. Sanity batch every ~100 GRPO steps (log 5 random rollouts).
6. Reward weighting in Rubric:
   - Programmatic (IoU, ID_r, ROUGE-L) → weight 1.0
   - LLMSim → weight 0.7
   - BERTScore → weight 0.5
7. Free-form length-grift coupling (DFT-C, HFE, HFD, QECC_65, GEO): geometric mean
   (ROUGE_Lsum/100)^0.6 * BERT_F1^0.4 replaces additive 0.5·ROUGE_L + 0.5·BERT_F1.
   BERT_F1 is the Curie cell 20 default (raw, no baseline rescale) — the earlier
   `rescale_with_baseline=True` deviation was reverted with W&B evidence (see
   the BERTScore entry under "Documented Deviations from Curie release"). The
   `max(0.0, raw)` clamp at the consumption site in `_freeform_geometric_reward`
   is now a defensive no-op on raw BERT (which lives in [0, 1]) but is kept as
   explicit input-domain enforcement for `freeform_geometric`. ROUGE-L is
   computed with English stopword filtering (custom tokenizer in scorers.py)
   — drops the content-free baseline from ~13/100 to ~0/100. Zero on either
   component returns 0 (signal must propagate); out-of-[0,1] inputs raise.
   Reward func wired at weight=1.0; max free-form contribution remains 1.0 (parity with
   programmatic tasks). freeform_weight in safeguards.yaml is now a legacy field.

## Sub-Agent Policy (Automatic)
For any task with 2+ independent parts, act as orchestrator and spawn minimum single-responsibility sub-agents in parallel: brainstormer, reviewer, tester.

## Stage 0 Safeguards (Enforced Everywhere)
- sub_llm_max_turns: 1
- per-step token budget + sandbox limits (PrimeSandbox)
- Model must always return {"ready": True} in answer dict
- Strict train/val/test splits — zero leakage

See @README.md for full pipeline and metrics.
See config/safeguards.yaml for exact locked values.
