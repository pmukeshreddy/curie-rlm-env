# curie-rlm-env — Curie Benchmark + Recursive Language Models + DPPO+KL (GRPO family)

**100% private local research project.**  
Strictly uses **Prime Intellect stack only**:
- `verifiers` (RLMEnv + Rubric + PrimeSandbox)
- `prime-rl` (GRPO trainer)

**Policy model**: Qwen3.5-7B (or Qwen/Qwen3.5-7B-Instruct).  
**No** Environment Hub. **No** `prime env init`. Everything stays 100% local.

## Locked Pipeline (Stage 0–8 — non-negotiable)
**Stage 0**: Safeguards + config (max_recursive_depth=1, deterministic-first rewards)  
**Stage 1**: uv setup + Prime libraries  
**Stage 2**: CurieRLMEnv inheriting official RLMEnv (Prime Sandboxes handle recursion/REPL/long-context)  
**Stage 3**: Baseline evaluation (RLM vs non-RLM)  
**Stage 3.5**: Baseline eval — Qwen3.5-7B + CurieRLMEnv on test split, no training. Establishes per-task floor.  
**Stage 4 (SKIPPED)**: no public SFT trajectory data exists for RLM on Curie. Cold-start GRPO chosen.  
**Stage 5**: RL post-training with prime-rl (DPPO+KL loss, DR-GRPO advantages — prime-rl's default in the GRPO family)  
**Stage 5.5 (CONTINGENCY)**: RFT bootstrap from stalled phase1 checkpoint. Filters phase1's own rollouts above reward T (default 0.5), SFTs on them, resumes phase1 from RFT checkpoint. Run only if trigger fires. See docs/STAGE_5_5_CONTINGENCY.md.  
**Stage 6 (SKIPPED)**: Originally "Mixing + online refinement". No concrete deliverable — async off-policy correction is handled internally by prime-rl (max_async_level=2 default). No mixing needed since SFT was skipped at Stage 4.  
**Stage 7**: Final eval + ablations + report  
**Stage 8**: Optional hub push (only when you decide)

## Curie Metrics (Locked — Verbatim from Paper + curie_run_eval.ipynb)
Per-task routing (dispatcher in Rubric):
- DFT-S, DFT-P, MPV → LLMSim (CoT-prompted mAP / recall / F1)
- BIOGR → Intersection-over-Union (IoU)
- PDB → Identity ratio (ID_r from RCSB pairwise alignment)
- HFE, HFD, QECC, GEO → ROUGE-L (programmatic) + LMScore (3-point LLM judge)

**Headline metric**: Per-task normalized score → average across all 10 tasks (matches Figure 5 of the paper).  
Optional extra: pass@k (threshold-based on task-specific score).

**Reward Hacking Guards (Enforced)**:
- Freeze judge model (Gemini/Claude, different family from Qwen policy)
- Programmatic spot-checks on LLMSim
- Length penalty
- Frozen LLMSim/LMScore prompts from Curie repo
- Sanity batch every ~100 GRPO steps

## Quickstart — Open in Claude Code
```bash
mkdir curie-rlm-env && cd curie-rlm-env
uv init --name curie-rlm-env
uv add "verifiers>=0.1.12" "prime-rl>=0.5.0"
uv add --dev pytest black ruff pytest-cov
mkdir -p config .claude/agents
```

Open folder in **Claude Code** and type:  
**"Start Stage 0 following the full pipeline in README.md using sub-agents"**
