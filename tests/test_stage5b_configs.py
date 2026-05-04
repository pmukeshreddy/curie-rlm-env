"""Stage 5b — RL training pipeline configs + judge cache tests.

Static structure tests only (file presence, TOML field correctness via tomllib
parse, judge cache behavior). prime-rl is NOT installed in this repo; configs
are validated for shape, not against a live runtime.
"""
from __future__ import annotations

import py_compile
import stat
import tomllib
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS = _PROJECT_ROOT / "configs"
_SCRIPTS = _PROJECT_ROOT / "scripts"

_CFG_NAMES = (
    "curie_grpo_freeform.toml",
    "curie_grpo_retrieval.toml",
    "curie_grpo_geometric.toml",
)


def _load_toml(name: str) -> dict:
    return tomllib.loads((_CONFIGS / name).read_text())


def _stripped_code(text: str) -> str:
    """Return text with full-line comments removed (preserves inline mixed lines)."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


# ---------------------------------------------------------------------------
# Config presence + structure
# ---------------------------------------------------------------------------

def test_three_configs_exist():
    for name in _CFG_NAMES:
        assert (_CONFIGS / name).is_file(), f"{name} missing"


def test_freeform_config_has_5_env_blocks():
    cfg = _load_toml("curie_grpo_freeform.toml")
    envs = cfg["orchestrator"]["train"]["env"]
    task_ids = sorted(e["args"]["task_id"] for e in envs)
    assert task_ids == sorted(["DFT-C", "HFE", "HFD", "QECC_65", "GEO"])


def test_retrieval_config_has_3_env_blocks():
    cfg = _load_toml("curie_grpo_retrieval.toml")
    envs = cfg["orchestrator"]["train"]["env"]
    task_ids = sorted(e["args"]["task_id"] for e in envs)
    assert task_ids == sorted(["DFT-S", "DFT-P", "MPVE"])


def test_retrieval_rollouts_per_example_is_4():
    # Stage 5b judge bottleneck mitigation per Stage 5a §8
    cfg = _load_toml("curie_grpo_retrieval.toml")
    assert cfg["orchestrator"]["rollouts_per_example"] == 4


def test_geometric_config_has_2_env_blocks():
    cfg = _load_toml("curie_grpo_geometric.toml")
    envs = cfg["orchestrator"]["train"]["env"]
    task_ids = sorted(e["args"]["task_id"] for e in envs)
    assert task_ids == sorted(["BIOGR", "PDB"])


def test_no_classical_grpo_kl_coef_present():
    # prime-rl uses kl_tau / dppo_mask_high / dppo_mask_low / adv_tau, NOT
    # classical kl_coef. Verify across all 3 configs (non-comment code only).
    for name in _CFG_NAMES:
        text = (_CONFIGS / name).read_text()
        code = _stripped_code(text)
        assert "kl_coef" not in code, f"{name} has classical kl_coef (use kl_tau)"


def test_seq_len_accommodates_curie():
    # Curie inputs avg 15k words; seq_len must accommodate input + completion.
    for name in _CFG_NAMES:
        cfg = _load_toml(name)
        assert cfg.get("seq_len", 0) >= 16384


def test_activation_checkpointing_enabled():
    # Stage 5a §11 OOM mitigation for 15k-token inputs
    for name in _CFG_NAMES:
        cfg = _load_toml(name)
        assert cfg["model"]["ac"] is True


def test_wandb_project_is_curie_rlm():
    for name in _CFG_NAMES:
        cfg = _load_toml(name)
        assert cfg["wandb"]["project"] == "curie-rlm"


# ---------------------------------------------------------------------------
# Judge cache module
# ---------------------------------------------------------------------------

def test_judge_cache_module_imports():
    from curie_rlm_env import judge_cache
    assert callable(judge_cache.cached_llmsim)
    assert callable(judge_cache.cached_llmsim_sync)
    assert callable(judge_cache.clear_cache)
    assert callable(judge_cache.cache_size)


def test_judge_cache_hit_miss():
    from curie_rlm_env.judge_cache import (
        cached_llmsim_sync, clear_cache, cache_size,
    )

    clear_cache()
    assert cache_size() == 0

    call_count = [0]

    def stub_judge(prompt: str) -> str:
        call_count[0] += 1
        return '{"json_extracted_index": 0}'

    gt = {"foo": "bar"}
    pred = [{"foo": "bar"}]
    prompt = "test prompt"

    # Miss → judge called
    r1 = cached_llmsim_sync(stub_judge, gt, pred, prompt)
    assert call_count[0] == 1

    # Hit → judge NOT called again
    r2 = cached_llmsim_sync(stub_judge, gt, pred, prompt)
    assert call_count[0] == 1
    assert r1 == r2

    # Different gt → miss
    cached_llmsim_sync(stub_judge, {"foo": "different"}, pred, prompt)
    assert call_count[0] == 2

    # Different pred → miss
    cached_llmsim_sync(stub_judge, gt, [{"foo": "x"}], prompt)
    assert call_count[0] == 3

    clear_cache()
    assert cache_size() == 0


# ---------------------------------------------------------------------------
# Probe + run scripts
# ---------------------------------------------------------------------------

def test_probe_script_imports():
    """Probe script must be syntactically valid."""
    script_path = _SCRIPTS / "probe_rollout_timing.py"
    assert script_path.is_file()
    py_compile.compile(str(script_path), doraise=True)


def test_run_scripts_exist_and_executable():
    for phase in (1, 2, 3):
        path = _SCRIPTS / f"run_phase{phase}.sh"
        assert path.is_file(), f"run_phase{phase}.sh missing"
        mode = path.stat().st_mode
        assert mode & stat.S_IXUSR, f"run_phase{phase}.sh not user-executable"
