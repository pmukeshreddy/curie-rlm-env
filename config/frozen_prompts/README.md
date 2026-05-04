# Frozen Prompts (Reward-Hacking Guard #4)

Prompts in this directory must be copied **verbatim** from the official Curie repo (`curie_run_eval.ipynb`). Stage 0 ships placeholder subdirectories only — no prompt content is committed until the verbatim source is supplied. Stage 3 (baseline evaluation) is the gate that fills these.

- `llmsim/` — LLMSim CoT-prompted scorer prompt (used by retrieval tasks: DFT-S, DFT-P, DFT-C, MPV).
- `lm_score/` — LMScore 3-point LLM judge prompt (used by freeform tasks: HFE, HFD, QECC, GEO).

**Anti-hallucination rule (CLAUDE.md):** never write or paraphrase prompt content here. Copy verbatim only. If the source is not yet available, leave the directory empty (the `.gitkeep` files preserve the structure).
