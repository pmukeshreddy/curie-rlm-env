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


def test_no_config_or_src_uses_invalid_qwen3_5_model_id():
    src_dir = _PROJECT_ROOT / "src"
    targets = list(_CONFIGS.glob("*.toml")) + list(src_dir.rglob("*.py"))
    for path in targets:
        text = path.read_text()
        assert "Qwen/Qwen3.5" not in text, (
            f"{path.relative_to(_PROJECT_ROOT)} still references invalid Qwen/Qwen3.5* HF repo"
        )


def test_continual_grpo_configs_use_qwen3_8b():
    for name in _CFG_NAMES:
        cfg = _load_toml(name)
        assert cfg["model"]["name"] == "Qwen/Qwen3-8B", (
            f"{name} should default to Qwen/Qwen3-8B; got {cfg['model']['name']!r}"
        )


def test_baseline_eval_default_model_is_qwen3_8b():
    src = (_PROJECT_ROOT / "src" / "curie_rlm_env" / "baseline_eval.py").read_text()
    assert 'model: str = "Qwen/Qwen3-8B"' in src


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


# ---------------------------------------------------------------------------
# Local inference routing (no prime_tunnel)
# ---------------------------------------------------------------------------

import os
import sys

import pytest as _pytest


def test_continual_scripts_do_not_hard_require_prime_api_key():
    """No continual run script may hard-fail when PRIME_API_KEY is unset.

    Documentation comments mentioning PRIME_API_KEY (e.g. "PRIME_API_KEY is NOT
    required") are fine — what's forbidden is a guard that exits when it's missing.
    """
    forbidden_patterns = (
        '${PRIME_API_KEY:-}',          # bash empty-default expansion in a guard
        'PRIME_API_KEY env var required',
        'PRIME_API_KEY is required',
        'export PRIME_API_KEY=',
    )
    for continual_phase in (1, 2, 3):
        path = _SCRIPTS / f"run_continual_phase{continual_phase}.sh"
        text = path.read_text()
        for pattern in forbidden_patterns:
            assert pattern not in text, (
                f"{path.name} appears to require PRIME_API_KEY ({pattern!r})"
            )


def test_readme_documents_prime_api_key_as_not_used_for_local():
    """README must state that PRIME_API_KEY is not used by any local-training path."""
    readme = (_PROJECT_ROOT / "README.md").read_text()
    assert "PRIME_API_KEY" in readme, "README must mention PRIME_API_KEY status"
    not_used_phrases = (
        "PRIME_API_KEY` is not used",
        "PRIME_API_KEY is not used",
        "PRIME_API_KEY is not required",
        "does not require `PRIME_API_KEY`",
    )
    assert any(phrase in readme for phrase in not_used_phrases), (
        "README must explicitly state that PRIME_API_KEY is not used by local-training paths"
    )


def test_continual_scripts_set_local_interception_defaults():
    """Each continual script must export local-routing host/bind defaults."""
    for continual_phase in (1, 2, 3):
        text = (_SCRIPTS / f"run_continual_phase{continual_phase}.sh").read_text()
        assert "CURIE_LOCAL_INTERCEPTION_HOST" in text
        assert "CURIE_LOCAL_INTERCEPTION_BIND" in text


def _reset_routing_env(monkeypatch):
    for name in (
        "CURIE_USE_PRIME_TUNNEL",
        "CURIE_LOCAL_INTERCEPTION_URL",
        "CURIE_LOCAL_INTERCEPTION_HOST",
        "CURIE_LOCAL_INTERCEPTION_PORT",
        "CURIE_LOCAL_INTERCEPTION_BIND",
    ):
        monkeypatch.delenv(name, raising=False)


def test_resolve_local_interception_returns_local_mode_by_default(monkeypatch):
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.env import resolve_local_interception_settings

    _reset_routing_env(monkeypatch)
    settings = resolve_local_interception_settings()
    assert settings is not None
    assert settings["host"] == "127.0.0.1"
    assert settings["bind"] == "127.0.0.1"
    assert settings["auto_port"] is True
    assert settings["override_url"] is None  # built lazily after server bind


def test_resolve_local_interception_never_returns_none_with_prime_tunnel_set(monkeypatch):
    """Strict: CURIE_USE_PRIME_TUNNEL=1 is no longer an opt-out; routing stays local."""
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.env import resolve_local_interception_settings

    _reset_routing_env(monkeypatch)
    monkeypatch.setenv("CURIE_USE_PRIME_TUNNEL", "1")  # ignored — local routing is mandatory
    settings = resolve_local_interception_settings()
    assert settings is not None
    assert settings["host"] == "127.0.0.1"


def test_resolve_local_interception_pins_port_when_set(monkeypatch):
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.env import resolve_local_interception_settings

    _reset_routing_env(monkeypatch)
    monkeypatch.setenv("CURIE_LOCAL_INTERCEPTION_HOST", "10.0.0.5")
    monkeypatch.setenv("CURIE_LOCAL_INTERCEPTION_PORT", "9099")
    settings = resolve_local_interception_settings()
    assert settings is not None
    assert settings["host"] == "10.0.0.5"
    assert settings["port"] == 9099
    assert settings["auto_port"] is False
    assert settings["override_url"] == "http://10.0.0.5:9099"


def test_resolve_local_interception_explicit_url_takes_precedence(monkeypatch):
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.env import resolve_local_interception_settings

    _reset_routing_env(monkeypatch)
    monkeypatch.setenv("CURIE_LOCAL_INTERCEPTION_URL", "http://my-host:7777/")
    monkeypatch.setenv("CURIE_LOCAL_INTERCEPTION_HOST", "ignored.example")
    monkeypatch.setenv("CURIE_LOCAL_INTERCEPTION_PORT", "1234")
    settings = resolve_local_interception_settings()
    assert settings is not None
    assert settings["override_url"] == "http://my-host:7777/"
    assert settings["host"] == "my-host"
    assert settings["port"] == 7777


def test_resolve_local_interception_url_without_port_rejected(monkeypatch):
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.env import resolve_local_interception_settings

    _reset_routing_env(monkeypatch)
    monkeypatch.setenv("CURIE_LOCAL_INTERCEPTION_URL", "http://my-host")
    with _pytest.raises(ValueError):
        resolve_local_interception_settings()


def test_check_local_inference_routing_script_compiles():
    path = _SCRIPTS / "check_local_inference_routing.py"
    assert path.is_file()
    py_compile.compile(str(path), doraise=True)


def test_continual_configs_still_use_qwen3_8b_after_routing_fix():
    """Routing fix must not change the Qwen/Qwen3-8B default model id."""
    for name in _CFG_NAMES:
        cfg = _load_toml(name)
        assert cfg["model"]["name"] == "Qwen/Qwen3-8B"


def test_continual_configs_still_have_single_env_after_routing_fix():
    for name in _CFG_NAMES:
        cfg = _load_toml(name)
        envs = cfg["orchestrator"]["train"]["env"]
        assert len(envs) == 1
        assert envs[0]["args"].get("continual_phase") in {1, 2, 3}
        assert "task_id" not in envs[0]["args"]


def test_trainer_model_ac_freq_one_still_present_after_routing_fix():
    for name in _CFG_NAMES:
        cfg = _load_toml(name)
        assert cfg["trainer"]["model"]["ac"] == {"freq": 1}


# ---------------------------------------------------------------------------
# Local Docker sandbox backend
# ---------------------------------------------------------------------------


def _reset_sandbox_env(monkeypatch):
    monkeypatch.delenv("CURIE_SANDBOX_BACKEND", raising=False)
    monkeypatch.delenv("CURIE_SANDBOX_NETWORK", raising=False)


def test_resolve_sandbox_backend_defaults_to_local_docker(monkeypatch):
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.local_sandbox import resolve_sandbox_backend

    _reset_sandbox_env(monkeypatch)
    assert resolve_sandbox_backend() == "local_docker"


def test_resolve_sandbox_backend_only_accepts_local_docker(monkeypatch):
    """Strict: 'prime' is no longer a valid backend; only local_docker is supported."""
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.local_sandbox import resolve_sandbox_backend

    _reset_sandbox_env(monkeypatch)
    for invalid in ("prime", "subprocess", "remote", "hosted"):
        monkeypatch.setenv("CURIE_SANDBOX_BACKEND", invalid)
        with _pytest.raises(ValueError):
            resolve_sandbox_backend()


def test_local_docker_sandbox_client_implements_required_interface():
    """LocalDockerSandboxClient must expose every method SandboxMixin awaits."""
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.local_sandbox import LocalDockerSandboxClient

    required = (
        "create",
        "wait_for_creation",
        "delete",
        "bulk_delete",
        "execute_command",
        "upload_file",
        "download_file",
        "read_file",
        "run_background_job",
        "teardown",
    )
    client = LocalDockerSandboxClient()
    for name in required:
        assert hasattr(client, name), f"LocalDockerSandboxClient missing {name!r}"


def test_local_docker_sandbox_client_lazy_imports_docker():
    """Constructing the client must NOT touch the Docker daemon (lazy)."""
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.local_sandbox import LocalDockerSandboxClient

    # Should construct without docker installed/running. This test passes regardless of
    # docker availability — the assertion is "no exception during __init__".
    client = LocalDockerSandboxClient()
    # Internal state: no docker client yet.
    assert client._docker is None
    assert client._containers == {}


def test_local_docker_rlm_executor_subclasses_rlm_executor():
    """LocalDockerRLMExecutor must be a real subclass of verifiers RLMExecutor."""
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.local_executor import LocalDockerRLMExecutor
    from verifiers.envs.experimental.rlm_env import RLMExecutor

    assert issubclass(LocalDockerRLMExecutor, RLMExecutor)


def test_local_docker_rlm_executor_overrides_teardown_paths():
    """LocalDockerRLMExecutor must override every teardown method that constructs
    prime_sandboxes.SandboxClient(APIClient()) inline upstream."""
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.local_executor import LocalDockerRLMExecutor

    for name in ("__init__", "teardown_sandboxes", "teardown"):
        method = LocalDockerRLMExecutor.__dict__.get(name)
        assert method is not None, (
            f"LocalDockerRLMExecutor must override {name!r} to keep traffic off "
            "prime_sandboxes — that's the whole point of the subclass"
        )


def test_local_docker_rlm_executor_does_not_import_prime_sandboxes():
    """Source-level audit: the subclass must not import from prime_sandboxes.

    Without that import, the upstream `SandboxClient(APIClient())` construction
    pattern is impossible by construction — that's the whole point of the
    subclass and a stronger guarantee than a substring grep against the file body.
    """
    import ast

    src = (
        _PROJECT_ROOT / "src" / "curie_rlm_env" / "local_executor.py"
    ).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "prime_sandboxes", (
                f"local_executor.py imports from prime_sandboxes at line {node.lineno}"
            )
            assert not (node.module or "").startswith("prime_sandboxes."), (
                f"local_executor.py imports submodule of prime_sandboxes at line {node.lineno}"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "prime_sandboxes", (
                    f"local_executor.py imports prime_sandboxes at line {node.lineno}"
                )


def test_curie_rlm_env_uses_local_executor_after_super_init():
    """Source-level audit: CurieRLMEnv.__init__ must replace self._executor with
    LocalDockerRLMExecutor after super().__init__()."""
    src = (_PROJECT_ROOT / "src" / "curie_rlm_env" / "env.py").read_text()
    assert "self._executor = LocalDockerRLMExecutor(self)" in src


def test_continual_scripts_default_sandbox_to_local_docker():
    """Each continual script must export CURIE_SANDBOX_BACKEND with local_docker default."""
    for continual_phase in (1, 2, 3):
        text = (_SCRIPTS / f"run_continual_phase{continual_phase}.sh").read_text()
        assert "CURIE_SANDBOX_BACKEND" in text
        assert "local_docker" in text


def test_check_local_runtime_script_compiles():
    path = _SCRIPTS / "check_local_runtime.py"
    assert path.is_file()
    py_compile.compile(str(path), doraise=True)


def test_readme_documents_local_docker_default():
    readme = (_PROJECT_ROOT / "README.md").read_text()
    assert "local_docker" in readme
    assert "CURIE_SANDBOX_BACKEND" in readme
    assert "PRIME_API_KEY" in readme


def test_continual_design_unchanged_after_local_sandbox():
    """Sandbox backend swap must not touch continual replay design.

    Re-asserts the four invariants the user specified: continual_phase, replay
    mixture, Qwen/Qwen3-8B, and the [trainer.model.ac] freq=1 line.
    """
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.continual import CONTINUAL_PHASES, mixture_for_continual_phase

    # Continual phases unchanged.
    assert set(CONTINUAL_PHASES.keys()) == {1, 2, 3}
    # Replay mixture unchanged: Phase 1 100% current; Phase 2 70/30; Phase 3 60/20/20.
    assert mixture_for_continual_phase(1) == {"current": 1.0}
    assert mixture_for_continual_phase(2) == {"current": 0.70, "phase1": 0.30}
    assert mixture_for_continual_phase(3) == {"current": 0.60, "phase1": 0.20, "phase2": 0.20}

    # Model + activation checkpointing unchanged.
    for name in _CFG_NAMES:
        cfg = _load_toml(name)
        assert cfg["model"]["name"] == "Qwen/Qwen3-8B"
        assert cfg["trainer"]["model"]["ac"] == {"freq": 1}
        envs = cfg["orchestrator"]["train"]["env"]
        assert len(envs) == 1
        assert envs[0]["args"].get("continual_phase") in {1, 2, 3}


def test_curie_rubric_signature_unchanged():
    """The CurieRubric class shape (judge_client kwarg) must remain stable."""
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.rubric import CurieRubric
    import inspect

    params = inspect.signature(CurieRubric.__init__).parameters
    assert "judge_client" in params, "CurieRubric.__init__ must accept judge_client"


# ---------------------------------------------------------------------------
# Strict: no opt-out paths anywhere in source / scripts / docs
# ---------------------------------------------------------------------------


_AUDITED_DIRS = ("configs", "scripts", "src", "tests")


def _audit_files() -> list[Path]:
    files: list[Path] = []
    for d in _AUDITED_DIRS:
        files.extend((_PROJECT_ROOT / d).rglob("*.py"))
        files.extend((_PROJECT_ROOT / d).rglob("*.sh"))
        files.extend((_PROJECT_ROOT / d).rglob("*.toml"))
    files.append(_PROJECT_ROOT / "README.md")
    return [p for p in files if "__pycache__" not in p.parts]


# Test files are allowed to mention these tokens in NEGATIVE assertions; we still
# check production code (src + scripts + configs + docs) is clean.
_PROD_DIRS = ("configs", "scripts", "src")


def _prod_files() -> list[Path]:
    files: list[Path] = []
    for d in _PROD_DIRS:
        files.extend((_PROJECT_ROOT / d).rglob("*.py"))
        files.extend((_PROJECT_ROOT / d).rglob("*.sh"))
        files.extend((_PROJECT_ROOT / d).rglob("*.toml"))
    files.append(_PROJECT_ROOT / "README.md")
    return [p for p in files if "__pycache__" not in p.parts]


def test_no_curie_use_prime_tunnel_anywhere():
    """The CURIE_USE_PRIME_TUNNEL opt-out env var was removed; no source may reference it."""
    offenders: list[str] = []
    for path in _prod_files():
        if "CURIE_USE_PRIME_TUNNEL" in path.read_text():
            offenders.append(str(path.relative_to(_PROJECT_ROOT)))
    assert not offenders, (
        f"CURIE_USE_PRIME_TUNNEL still referenced in production paths: {offenders}"
    )


def test_no_prod_source_documents_prime_api_key_as_opt_in():
    """Production code/docs must not present PRIME_API_KEY as an opt-in path.

    Allowed: README documenting it as 'not used'; comments quoting upstream
    error strings (e.g. the verifiers/prime_tunnel error text). Forbidden:
    any line that suggests setting PRIME_API_KEY enables a working code path.
    """
    forbidden_phrases = (
        "set CURIE_USE_PRIME_TUNNEL",
        "set CURIE_SANDBOX_BACKEND=prime",
        "opt into the hosted",
        "opt back into the hosted",
        "PRIME_API_KEY is REQUIRED",
        "PRIME_API_KEY is required",
    )
    offenders: list[tuple[str, str]] = []
    for path in _prod_files():
        text = path.read_text()
        for phrase in forbidden_phrases:
            if phrase.lower() in text.lower():
                offenders.append((str(path.relative_to(_PROJECT_ROOT)), phrase))
    assert not offenders, (
        f"Production paths still document an opt-in to Prime hosted services: {offenders}"
    )


def test_no_affirmative_fallback_phrasing_in_prod_source():
    """Strict-failure audit: production source must not advertise fallback paths.

    Affirmative patterns ('falls back to', 'legacy behavior', 'backward-compat
    fallback', 'fallback path') describe a path that exists. Strict-failure
    documentation ('ZERO-FALLBACK', 'no fallback', 'never fall back') is fine
    — those are explicit statements that no such path exists, which is the
    audit's whole point. The test forbids the former and allows the latter.
    """
    forbidden = (
        "falls back to",
        "fall back to",
        "fallback path",
        "fallback behavior",
        "legacy behavior",
        "legacy fallback",
        "backward-compat fallback",
        "backward compat fallback",
    )
    offenders: list[tuple[str, str]] = []
    for path in (_PROJECT_ROOT / "src").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text().lower()
        for phrase in forbidden:
            if phrase in text:
                offenders.append((str(path.relative_to(_PROJECT_ROOT)), phrase))
    assert not offenders, (
        f"Production src/ still contains fallback advertisement language: {offenders}"
    )


def test_resolve_local_interception_rejects_url_without_port(monkeypatch):
    """Misconfigured URL (no port) must raise — no silent repair to a default port."""
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from curie_rlm_env.env import resolve_local_interception_settings

    _reset_routing_env(monkeypatch)
    monkeypatch.setenv("CURIE_LOCAL_INTERCEPTION_URL", "http://my-host")
    with _pytest.raises(ValueError):
        resolve_local_interception_settings()
