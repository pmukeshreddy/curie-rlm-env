# curie-rlm-env — CURIE Benchmark + RLM + Continual DPPO+KL

Quote from `src/curie_rlm_env/continual.py`:
`Phase 2: 70% retrieval current tasks + 30% Phase 1 replay.`

Private local research project implementing Google CURIE scientific long-context tasks with Recursive Language Models. The default policy model is `Qwen/Qwen3-8B`. The earlier `Qwen/Qwen3.5-7B-Instruct` was removed because it is not a valid Hugging Face repo (no Qwen3.5 family exists on HF — only Qwen2.5 and Qwen3); override the default at the prime-rl CLI with `--model.name <hf_repo>` if needed. The environment inherits the official `verifiers.envs.experimental.rlm_env.RLMEnv`; recursion, long-context execution, and sandboxed code run through Prime/verifiers.

## Local runtime (no Prime hosted services required)

Local single-pod training does NOT require `PRIME_API_KEY`. `CurieRLMEnv` runs sandboxes on the **local Docker daemon** by default and resolves a sandbox→env-worker callback URL locally — both the `prime_sandboxes` hosted backend and the `prime_tunnel` hosted tunnel are bypassed. Since the GPU pod is already paid for, this adds no separate Prime-billed sandbox usage.

Required env vars (local training):

| Variable | Purpose | Required for |
|---|---|---|
| `INFERENCE_SERVER_IP` | Address of the local prime-rl inference server | Stage 5 continual training |
| `HF_TOKEN` | Hugging Face auth for `Qwen/Qwen3-8B` download | First-time model fetch |
| `WANDB_API_KEY` | W&B logging | Stage 5 (if W&B enabled) |
| `GEMINI_API_KEY` | LLMSim judge for retrieval rewards | Continual Phases 2 and 3 only |

NOT required for local training:

| Variable | Purpose |
|---|---|
| `PRIME_API_KEY` | Prime hosted tunnel + hosted sandbox — only needed when `CURIE_USE_PRIME_TUNNEL=1` or `CURIE_SANDBOX_BACKEND=prime` |

Optional local-routing knobs (all have defaults):

| Variable | Default | Purpose |
|---|---|---|
| `CURIE_SANDBOX_BACKEND` | `local_docker` | `local_docker` (run sandboxes via local Docker daemon) or `prime` (use hosted Prime sandboxes; requires `PRIME_API_KEY`) |
| `CURIE_SANDBOX_NETWORK` | `bridge` | Docker network mode for sandbox containers (`bridge`, `host`, custom network name) |
| `CURIE_LOCAL_INTERCEPTION_HOST` | `127.0.0.1` | Host portion of the sandbox→env-worker callback URL |
| `CURIE_LOCAL_INTERCEPTION_PORT` | auto-assigned | Pin a fixed port; otherwise the URL is built after the interception server binds |
| `CURIE_LOCAL_INTERCEPTION_BIND` | `127.0.0.1` | Bind interface; set `0.0.0.0` if the sandbox runs in a separate netns (e.g. docker bridge) |
| `CURIE_LOCAL_INTERCEPTION_URL` | (composed) | Full override URL; takes precedence over HOST/PORT |
| `INFERENCE_SERVER_API_KEY` | (none) | Local auth secret if your local inference server requires one |
| `CURIE_USE_PRIME_TUNNEL` | unset | Set to `1` to opt into the hosted-tunnel path (requires `PRIME_API_KEY`) |

Local Docker sandbox prerequisites:
- Docker daemon reachable from the trainer process (e.g. `/var/run/docker.sock` mounted, or `DOCKER_HOST` set).
- The `docker` Python SDK installed (`uv pip install docker`).
- The image referenced in `safeguards.yaml` / `CreateSandboxRequest` is pullable (default `python:3.11-slim`).

Verify routing:

```bash
PYTHONPATH=/workspace/curie-rlm-env/src \
    uv run --project /workspace/prime-rl python scripts/check_local_runtime.py
```

Exits 0 when local mode is healthy (Docker reachable, backend=local_docker, interception local), 1 otherwise. The earlier `scripts/check_local_inference_routing.py` checks only the interception path and is still available for narrower diagnosis.

## Locked Stack

- `verifiers` for `RLMEnv`, `RubricGroup`, `JudgeRubric`, `SandboxMixin`, and RLM monitor rubrics
- `prime-rl` installed separately on the GPU pod for Stage 5 training
- Python `~=3.12.0`
- `Qwen/Qwen3-8B` policy model (default; override at the prime-rl CLI with `--model.name <hf_repo>`)
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

Sequential family-only training has been replaced. Training now uses one continual replay environment per continual phase:

| Phase | Current Tasks | Replay Tasks | Mixture |
|---|---|---|---|
| 1 | DFT-C, HFE, HFD, QECC_65, GEO | none | 100% current |
| 2 | DFT-S, DFT-P, MPVE | Phase 1 tasks | 70% current, 30% Phase 1 replay |
| 3 | BIOGR, PDB | Phase 1 tasks and Phase 2 tasks | 60% current, 20% Phase 1 replay, 20% Phase 2 replay |

The training configs call `CurieRLMEnv` with `continual_phase = 1`, `continual_phase = 2`, or `continual_phase = 3`. Single-task `task_id` loading remains for eval and rubric compatibility, but it is not the Stage 5 training path.

Run order:

```bash
./scripts/run_continual_phase1.sh

export CONTINUAL_PHASE1_CKPT=outputs/continual_phase1/step_<N>/
./scripts/run_continual_phase2.sh

export CONTINUAL_PHASE2_CKPT=outputs/continual_phase2/step_<N>/
./scripts/run_continual_phase3.sh
```

Phase 2 and Phase 3 require `GEMINI_API_KEY` because retrieval tasks are present through current tasks or replay. `CurieRLMEnv` wires that key into `CurieRubric` as the LLMSim judge client, and both scripts export `CURIE_JUDGE_CACHE=1`.

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
- `scripts/eval_retention_forgetting.py`
