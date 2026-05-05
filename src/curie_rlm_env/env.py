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
from .local_executor import LocalDockerRLMExecutor
from .local_sandbox import _dbg, resolve_sandbox_backend
from .rubric import CurieRubric
from .schema import validate_answer

_CFG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "safeguards.yaml"
)

# Local-inference routing env vars. Setting `_interception_url_override` on the RLMEnv
# bypasses the prime_tunnel.Tunnel code path at verifiers/envs/experimental/rlm_env.py:3445
# (which raises `TunnelError("No API key configured. Set PRIME_API_KEY environment variable.")`
# from prime_tunnel/core/client.py:75). Local routing is mandatory; there is no opt-out.
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


def resolve_local_interception_settings() -> _LocalInterceptionSettings:
    """Resolve sandbox→env-worker callback URL for local-only training.

    Always returns a settings dict (never None). Raises ValueError on misconfigured
    env vars — there is no opt-out path back to the prime_tunnel hosted service.

    Resolution order:
      1. CURIE_LOCAL_INTERCEPTION_URL=http://host:port  → use that exact URL, pin port
      2. CURIE_LOCAL_INTERCEPTION_PORT=N                → http://HOST:N, pin port
      3. (no port set)                                   → auto-assign port; URL is
         http://HOST:<actual_port> after the interception server binds.

    HOST defaults to 127.0.0.1; BIND defaults to 127.0.0.1 (set 0.0.0.0 if the sandbox
    runs in a separate network namespace, e.g. docker bridge).
    """
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
        self._curie_local_host = settings["host"]
        self._curie_local_auto_port = settings["auto_port"]
        self._interception_bind_host = settings["bind"]
        self.interception_port = settings["port"]
        if settings["override_url"] is not None:
            self._interception_url_override = settings["override_url"]

        # Sandbox backend resolution is strict: only local_docker is valid; anything
        # else (including unset → defaulted) only ever returns "local_docker".
        # The actual sandbox client lives on RLMExecutor (a SandboxMixin), not on
        # the env; replace the executor wholesale with a subclass that wires
        # LocalDockerSandboxClient through every code path (incl. the teardown
        # methods that would otherwise construct prime_sandboxes.SandboxClient
        # inline). See src/curie_rlm_env/local_executor.py for the design notes.
        self._curie_sandbox_backend = resolve_sandbox_backend()
        if not hasattr(self, "_executor"):
            raise RuntimeError(
                "CurieRLMEnv: RLMEnv.__init__ did not set self._executor; "
                "verifiers RLMExecutor wiring has changed."
            )
        self._executor.sandbox_client.teardown(wait=False)
        self._executor = LocalDockerRLMExecutor(self)
        _dbg(
            f"CurieRLMEnv ready task_id={self.task_id} continual_phase={self.continual_phase} "
            f"interception_url_override={self._interception_url_override!r} "
            f"interception_port={self.interception_port} "
            f"interception_bind={self._interception_bind_host} "
            f"sandbox_client={type(self._executor.sandbox_client).__name__}"
        )

    async def _setup_interception_and_register(
        self, state: State, rollout_id: str
    ) -> State:
        if self._curie_local_auto_port and not self._interception_url_override:
            await self._ensure_interception_server()
            self._interception_url_override = (
                f"http://{self._curie_local_host}:{self.interception_port}"
            )
            _dbg(
                f"interception URL pinned after server bind: "
                f"{self._interception_url_override} (rollout_id={rollout_id})"
            )
        return await super()._setup_interception_and_register(state, rollout_id)

    @vf.stop
    async def answer_schema_valid(self, state: State) -> bool:
        has_final = "final_answer" in state
        ans = state.get("final_answer")
        ans_repr = (ans[:120] + "…") if isinstance(ans, str) and len(ans) > 120 else repr(ans)
        traj = state.get("trajectory") or []
        # Surface the rollout-level signals that explain WHY a trajectory might be empty.
        # `prompt_too_long` and `is_truncated` are set by upstream RLMEnv; the
        # root_llm_* and *_call_count fields tell us whether the root model and tools
        # ran at all. `error` is the upstream rollout error (if any).
        _dbg(
            f"answer_schema_valid rollout_id={state.get('rollout_id')!r} "
            f"has_final_answer={has_final} answer={ans_repr} "
            f"trajectory_turns={len(traj)} "
            f"root_llm_turns={state.get('root_llm_turns')!r} "
            f"root_llm_prompt_tokens={state.get('root_llm_prompt_tokens')!r} "
            f"root_llm_completion_tokens={state.get('root_llm_completion_tokens')!r} "
            f"root_tool_call_count={state.get('root_tool_call_count')!r} "
            f"sub_llm_call_count={state.get('sub_llm_call_count')!r} "
            f"sub_llm_total_turns={state.get('sub_llm_total_turns')!r} "
            f"is_completed={state.get('is_completed')!r} "
            f"is_truncated={state.get('is_truncated')!r} "
            f"prompt_too_long={state.get('prompt_too_long')!r} "
            f"max_turns_in_context_stopped={state.get('max_turns_in_context_stopped')!r} "
            f"error={state.get('error')!r} "
            f"final_env_response={(repr(state.get('final_env_response'))[:200])!r}"
        )
        if not has_final:
            return False
        try:
            validate_answer(state["final_answer"])
        except ValueError as exc:
            _dbg(f"answer_schema_valid REJECT rollout_id={state.get('rollout_id')!r} reason={exc}")
            raise
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
