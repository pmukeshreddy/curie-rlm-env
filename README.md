# curie-rlm-env — CURIE Benchmark + RLM + Continual DPPO+KL

Quote from `src/curie_rlm_env/continual.py`:
`Phase 2: 70% retrieval current tasks + 30% Phase 1 replay.`

Private local research project implementing Google CURIE scientific long-context tasks with Recursive Language Models. The policy model is Qwen3.5-7B. The environment inherits the official `verifiers.envs.experimental.rlm_env.RLMEnv`; recursion, long-context execution, and sandboxed code run through Prime/verifiers.

## Locked Stack

- `verifiers` for `RLMEnv`, `RubricGroup`, `JudgeRubric`, `SandboxMixin`, and RLM monitor rubrics
- `prime-rl` installed separately on the GPU pod for Stage 5 training
- Python `~=3.12.0`
- Qwen3.5-7B policy model
- No Environment Hub dependency for local research runs

## Pipeline

**Stage 0**: Safeguards and locked config values  
**Stage 1**: uv setup and Prime/verifiers imports  
**Stage 2**: `CurieRLMEnv` extending official `RLMEnv`  
**Stage 3**: CURIE rubric dispatcher and scorers  
**Stage 3.5**: Baseline eval on held-out splits, no training  
**Stage 4 (SKIPPED)**: no public SFT trajectory data exists for RLM on CURIE  
**Stage 5**: continual replay RL with prime-rl DPPO+KL and DR-GRPO advantages  
**Stage 5.5 (CONTINGENCY)**: RFT bootstrap from stalled continual Phase 1 rollouts only  
**Stage 6 (SKIPPED)**: no separate mixing stage; replay is now inside Stage 5 training  
**Stage 6.5**: verification  
**Stage 7**: internal final eval, ablations, report  
**Stage 8**: optional hub push

## Continual Stage 5

Sequential family-only training has been replaced. Training now uses one continual replay environment per phase:

| Phase | Current Tasks | Replay Tasks | Mixture |
|---|---|---|---|
| 1 | DFT-C, HFE, HFD, QECC_65, GEO | none | 100% current |
| 2 | DFT-S, DFT-P, MPVE | Phase 1 tasks | 70% current, 30% Phase 1 replay |
| 3 | BIOGR, PDB | Phase 1 tasks and Phase 2 tasks | 60% current, 20% Phase 1 replay, 20% Phase 2 replay |

The training configs call `CurieRLMEnv` with `phase = 1`, `phase = 2`, or `phase = 3`. Single-task `task_id` loading remains for eval and rubric compatibility, but it is not the Stage 5 training path.

Run order:

```bash
./scripts/run_continual_phase1.sh

export CONTINUAL_PHASE1_CKPT=outputs/continual_phase1/step_<N>/
./scripts/run_continual_phase2.sh

export CONTINUAL_PHASE2_CKPT=outputs/continual_phase2/step_<N>/
./scripts/run_continual_phase3.sh
```

Phase 2 and Phase 3 require `GEMINI_API_KEY` because retrieval tasks are present through current tasks or replay. Both scripts export `CURIE_JUDGE_CACHE=1`.

## CURIE Metrics

Rubric dispatch:

- Retrieval: DFT-S, DFT-P, MPVE → LLMSim precision/recall/F1
- Geometric: BIOGR → IoU
- Structural: PDB → identity ratio `ID_r` through FASTA `>` extraction only
- Free-form: DFT-C, HFE, HFD, QECC_65, GEO → ROUGE-L + BERTScore

Headline metric: per-task normalized score averaged across all 10 tasks.

## Key Files

- `src/curie_rlm_env/continual.py`: task groups, phase definitions, replay ratios, deterministic dataset mixing
- `src/curie_rlm_env/env.py`: `CurieRLMEnv` wiring for continual training phases and single-task eval
- `configs/curie_grpo_continual_phase1.toml`
- `configs/curie_grpo_continual_phase2.toml`
- `configs/curie_grpo_continual_phase3.toml`
- `scripts/run_continual_phase1.sh`
- `scripts/run_continual_phase2.sh`
- `scripts/run_continual_phase3.sh`
