"""CurieRLMEnv — Stage 2 wiring layer for CURIE benchmark.

Inherits verifiers.envs.experimental.rlm_env.RLMEnv. Reads safeguards from
config/safeguards.yaml and passes verbatim-named kwargs to super().__init__().

Stage 3b: vf.Rubric() placeholder replaced with CurieRubric() per-task
dispatcher. Note: CurieRubric judge_client defaults to None — production code
that needs LLMSim must construct CurieRubric directly with a real judge.

is_completed cannot be overridden (it is @final at environment.py:658). Schema
validation is wired via a @vf.stop-decorated method that returns True (signal
stop) or raises ValueError (loud schema fail). Returns False ONLY when the
final answer is not yet present in state — the multiturn-stop convention.

Training quote from src/curie_rlm_env/continual.py:
"Phase 2: 70% retrieval current tasks + 30% Phase 1 replay."
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TypedDict
from urllib.parse import urlparse

import yaml
import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import State

from .continual import CONTINUAL_SEED, load_continual_phase_dataset
from .datasets import load_curie_task
from .judge import make_gemini_judge_from_env
from .local_sandbox import (
    LocalDockerSandboxClient,
    resolve_sandbox_backend,
)
from .rubric import CurieRubric
from .schema import validate_answer

_CFG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "safeguards.yaml"
)

# Local-inference routing env vars (bypass prime_tunnel for single-pod local training).
# RLMEnv falls back to prime_tunnel.Tunnel() when its `_interception_url_override` is
# unset (verifiers/envs/experimental/rlm_env.py:3445), which raises
#   TunnelError("No API key configured. Set PRIME_API_KEY environment variable.")
# from prime_tunnel/core/client.py:75. Setting the override skips that branch.
_USE_PRIME_TUNNEL_ENV = "CURIE_USE_PRIME_TUNNEL"
_INTERCEPTION_URL_ENV = "CURIE_LOCAL_INTERCEPTION_URL"
_INTERCEPTION_HOST_ENV = "CURIE_LOCAL_INTERCEPTION_HOST"
_INTERCEPTION_PORT_ENV = "CURIE_LOCAL_INTERCEPTION_PORT"
_INTERCEPTION_BIND_ENV = "CURIE_LOCAL_INTERCEPTION_BIND"

_DEFAULT_LOCAL_HOST = "127.0.0.1"
_DEFAULT_LOCAL_BIND = "127.0.0.1"


class _LocalInterceptionSettings(TypedDict):
    override_url: Optional[str]
    host: str
    port: int
    bind: str
    auto_port: bool


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_local_interception_settings() -> Optional[_LocalInterceptionSettings]:
    """Resolve sandbox→env-worker callback URL for local-only training.

    Returns None when CURIE_USE_PRIME_TUNNEL=1 (caller wants the original prime_tunnel
    behavior, which requires PRIME_API_KEY). Otherwise returns settings the caller
    applies to the RLMEnv instance to bypass the tunnel.

    Resolution order:
      1. CURIE_USE_PRIME_TUNNEL=1                       → None (opt out)
      2. CURIE_LOCAL_INTERCEPTION_URL=http://host:port  → use that exact URL, pin port
      3. CURIE_LOCAL_INTERCEPTION_PORT=N                → http://HOST:N, pin port
      4. (no port set)                                   → auto-assign port; URL is
         http://HOST:<actual_port> after the interception server binds.

    HOST defaults to 127.0.0.1; BIND defaults to 127.0.0.1 (set 0.0.0.0 if the sandbox
    runs in a separate network namespace, e.g. docker bridge).
    """
    if _truthy(os.environ.get(_USE_PRIME_TUNNEL_ENV)):
        return None
    bind = os.environ.get(_INTERCEPTION_BIND_ENV, _DEFAULT_LOCAL_BIND)
    explicit_url = os.environ.get(_INTERCEPTION_URL_ENV)
    if explicit_url:
        parsed = urlparse(explicit_url)
        if parsed.port is None:
            raise ValueError(
                f"{_INTERCEPTION_URL_ENV}={explicit_url!r} must include a port"
            )
        return {
            "override_url": explicit_url,
            "host": parsed.hostname or _DEFAULT_LOCAL_HOST,
            "port": parsed.port,
            "bind": bind,
            "auto_port": False,
        }
    host = os.environ.get(_INTERCEPTION_HOST_ENV, _DEFAULT_LOCAL_HOST)
    port_env = os.environ.get(_INTERCEPTION_PORT_ENV)
    if port_env:
        port = int(port_env)
        return {
            "override_url": f"http://{host}:{port}",
            "host": host,
            "port": port,
            "bind": bind,
            "auto_port": False,
        }
    return {
        "override_url": None,
        "host": host,
        "port": 0,
        "bind": bind,
        "auto_port": True,
    }


class CurieRLMEnv(RLMEnv):
    """CurieRLMEnv for continual training phases and single-task eval."""

    def __init__(
        self,
        task_id: str | None = None,
        split: str = "test",
        continual_phase: int | None = None,
        seed: int = CONTINUAL_SEED,
    ):
        if (task_id is None) == (continual_phase is None):
            raise ValueError(
                "CurieRLMEnv requires exactly one of task_id (single-task eval) "
                "or continual_phase (continual training)."
            )
        cfg = yaml.safe_load(_CFG_PATH.read_text())
        dataset = (
            load_continual_phase_dataset(continual_phase, split=split, seed=seed)
            if continual_phase is not None
            else load_curie_task(task_id, split)
        )
        judge_client = (
            make_gemini_judge_from_env()
            if continual_phase in {2, 3}
            else None
        )
        rubric = CurieRubric(judge_client=judge_client)
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            sub_llm_max_turns=cfg["rlm_env"]["sub_llm_max_turns"],
            sub_max_completion_tokens=cfg["rlm_env"]["sub_max_completion_tokens"],
            sandbox_timeout_minutes=cfg["sandbox"]["sandbox_timeout_minutes"],
            sandbox_memory_gb=cfg["sandbox"]["sandbox_memory_gb"],
            code_execution_timeout=cfg["sandbox"]["code_execution_timeout"],
            abort_on_code_timeout=cfg["sandbox"]["abort_on_code_timeout"],
        )
        self.task_id = task_id if task_id is not None else f"continual_phase_{continual_phase}"
        self.continual_phase = continual_phase
        self.seed = seed

        settings = resolve_local_interception_settings()
        if settings is None:
            self._curie_local_inference = False
            self._curie_local_host = _DEFAULT_LOCAL_HOST
            self._curie_local_auto_port = False
        else:
            self._curie_local_inference = True
            self._curie_local_host = settings["host"]
            self._curie_local_auto_port = settings["auto_port"]
            self._interception_bind_host = settings["bind"]
            self.interception_port = settings["port"]
            if settings["override_url"] is not None:
                self._interception_url_override = settings["override_url"]

        self._curie_sandbox_backend = resolve_sandbox_backend()
        if self._curie_sandbox_backend == "local_docker":
            existing = getattr(self, "sandbox_client", None)
            if existing is not None and hasattr(existing, "teardown"):
                try:
                    existing.teardown(wait=False)
                except Exception:
                    pass
            self.sandbox_client = LocalDockerSandboxClient()

    async def _setup_interception_and_register(
        self, state: State, rollout_id: str
    ) -> State:
        if (
            self._curie_local_inference
            and self._curie_local_auto_port
            and not self._interception_url_override
        ):
            await self._ensure_interception_server()
            self._interception_url_override = (
                f"http://{self._curie_local_host}:{self.interception_port}"
            )
        return await super()._setup_interception_and_register(state, rollout_id)

    @vf.stop
    async def answer_schema_valid(self, state: State) -> bool:
        if "final_answer" not in state:
            return False
        validate_answer(state["final_answer"])
        return True


def load_task_environment(task_id: str, split: str = "test") -> CurieRLMEnv:
    """Load a single-task CURIE environment for eval and rubric compatibility."""
    return CurieRLMEnv(task_id=task_id, split=split)


def load_continual_environment(
    continual_phase: int,
    split: str = "train",
    seed: int = CONTINUAL_SEED,
) -> CurieRLMEnv:
    """Load a continual replay training environment."""
    return CurieRLMEnv(continual_phase=continual_phase, split=split, seed=seed)


def load_environment(
    task_id: str | None = None,
    split: str = "test",
    continual_phase: int | None = None,
    seed: int = CONTINUAL_SEED,
) -> CurieRLMEnv:
    """Prime/verifiers entrypoint.

    Training configs pass continual_phase=<1|2|3>. Baseline/eval callers keep task_id.
    """
    if continual_phase is not None:
        return load_continual_environment(continual_phase=continual_phase, split=split, seed=seed)
    if task_id is None:
        raise ValueError("load_environment requires task_id for eval or continual_phase for continual training.")
    return load_task_environment(task_id=task_id, split=split)
