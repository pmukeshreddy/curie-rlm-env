"""Stage 5 — continual replay RL training configs + judge cache tests.

Quote from src/curie_rlm_env/continual.py:
"Phase 2: 70% retrieval current tasks + 30% Phase 1 replay."

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
    "curie_grpo_continual_phase1.toml",
    "curie_grpo_continual_phase2.toml",
    "curie_grpo_continual_phase3.toml",
)

_OLD_CFG_NAMES = (
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

def test_three_continual_configs_exist():
    for name in _CFG_NAMES:
        assert (_CONFIGS / name).is_file(), f"{name} missing"


def test_old_sequential_configs_removed():
    for name in _OLD_CFG_NAMES:
        assert not (_CONFIGS / name).exists(), f"{name} must not remain as a training side path"


def test_continual_phase1_config_uses_continual_phase_env():
    cfg = _load_toml("curie_grpo_continual_phase1.toml")
    envs = cfg["orchestrator"]["train"]["env"]
    assert len(envs) == 1
    assert envs[0]["args"] == {"continual_phase": 1, "split": "train", "seed": 42}


def test_continual_phase2_config_uses_continual_phase_env():
    cfg = _load_toml("curie_grpo_continual_phase2.toml")
    envs = cfg["orchestrator"]["train"]["env"]
    assert len(envs) == 1
    assert envs[0]["args"] == {"continual_phase": 2, "split": "train", "seed": 42}


def test_continual_phase3_config_uses_continual_phase_env():
    cfg = _load_toml("curie_grpo_continual_phase3.toml")
    envs = cfg["orchestrator"]["train"]["env"]
    assert len(envs) == 1
    assert envs[0]["args"] == {"continual_phase": 3, "split": "train", "seed": 42}


def test_training_configs_do_not_pass_single_task_ids():
    for name in _CFG_NAMES:
        cfg = _load_toml(name)
        envs = cfg["orchestrator"]["train"]["env"]
        assert all("task_id" not in env["args"] for env in envs)


def test_retrieval_replay_phases_rollouts_per_example_is_4():
    for name in ("curie_grpo_continual_phase2.toml", "curie_grpo_continual_phase3.toml"):
        cfg = _load_toml(name)
        assert cfg["orchestrator"]["rollouts_per_example"] == 4


def test_phase1_rollouts_per_example_is_16():
    cfg = _load_toml("curie_grpo_continual_phase1.toml")
    assert cfg["orchestrator"]["rollouts_per_example"] == 16


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


def test_no_config_uses_invalid_model_ac_true():
    invalid_model_ac = "ac" + " = " + "true"
    for path in _CONFIGS.glob("*.toml"):
        code = _stripped_code(path.read_text())
        assert invalid_model_ac not in code, f"{path.name} uses invalid Prime-RL model.ac"


def test_continual_activation_checkpointing_uses_trainer_model_ac():
    # Prime-RL expects activation checkpointing under trainer.model.ac.
    for name in _CFG_NAMES:
        cfg = _load_toml(name)
        assert "ac" not in cfg["model"]
        assert cfg["trainer"]["model"]["ac"] == {"freq": 1}


def test_wandb_project_is_curie_rlm():
    for name in _CFG_NAMES:
        cfg = _load_toml(name)
        assert cfg["wandb"]["project"] == "curie-rlm"


def test_datasets_declared_in_pyproject_dependencies():
    pyproject = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    assert any(dep.startswith("datasets>=") for dep in deps)


def test_google_genai_declared_for_gemini_judge():
    pyproject = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    assert any(dep.startswith("google-genai>=") for dep in deps)


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


def test_continual_run_scripts_exist_and_executable():
    for continual_phase in (1, 2, 3):
        path = _SCRIPTS / f"run_continual_phase{continual_phase}.sh"
        assert path.is_file(), f"run_continual_phase{continual_phase}.sh missing"
        mode = path.stat().st_mode
        assert mode & stat.S_IXUSR, f"run_continual_phase{continual_phase}.sh not user-executable"


def test_old_sequential_run_scripts_removed():
    for continual_phase in (1, 2, 3):
        path = _SCRIPTS / f"run_phase{continual_phase}.sh"
        assert not path.exists(), f"{path.name} must not remain as a training side path"


def test_continual_replay_scripts_use_new_checkpoint_vars():
    phase2 = (_SCRIPTS / "run_continual_phase2.sh").read_text()
    phase3 = (_SCRIPTS / "run_continual_phase3.sh").read_text()
    assert "CONTINUAL_PHASE1_CKPT" in phase2
    assert "CONTINUAL_PHASE2_CKPT" in phase3
    assert '"$PHASE1_CKPT"' not in phase2
    assert '"$PHASE2_CKPT"' not in phase3


def test_retrieval_replay_scripts_enable_judge_cache():
    for continual_phase in (2, 3):
        text = (_SCRIPTS / f"run_continual_phase{continual_phase}.sh").read_text()
        assert "GEMINI_API_KEY" in text
        assert "CURIE_JUDGE_CACHE=1" in text
