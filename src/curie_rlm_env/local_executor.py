"""Local-Docker-backed RLMExecutor for CurieRLMEnv.

Subclasses verifiers' ``RLMExecutor`` so that **every** sandbox code path —
create, execute, teardown — goes through ``LocalDockerSandboxClient`` instead
of ``prime_sandboxes``. Replacing the executor is the only way to hit all the
paths cleanly:

  * ``RLMExecutor.__init__`` (rlm_env.py:1478) inherits ``SandboxMixin`` and
    calls ``init_sandbox_client``, which sets ``self.sandbox_client`` to a
    ``ThreadedAsyncSandboxClient`` that proxies to ``prime_sandboxes``. We
    swap that immediately after super().__init__().
  * ``SandboxMixin.teardown_sandboxes`` (sandbox_mixin.py:451) and
    ``RLMExecutor.teardown`` (rlm_env.py:1656) both construct
    ``SandboxClient(APIClient())`` *inline* — the only way to stop those paths
    from reaching the hosted API is to override the methods. Both are
    documented subclass extension points (see ``teardown_mixin_sandboxes``
    docstring at sandbox_mixin.py:472).

CurieRLMEnv constructs an instance of this class and assigns it to
``self._executor`` after the parent ``RLMEnv.__init__`` completes.
"""
from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from verifiers.envs.experimental.rlm_env import RLMExecutor

from .local_sandbox import LocalDockerSandboxClient

if TYPE_CHECKING:
    from verifiers.envs.experimental.rlm_env import RLMEnv


class LocalDockerRLMExecutor(RLMExecutor):
    """RLMExecutor wired to LocalDockerSandboxClient on every code path."""

    def __init__(self, env: "RLMEnv") -> None:
        super().__init__(env)
        # SandboxMixin.init_sandbox_client just gave us a ThreadedAsyncSandboxClient
        # that proxies to prime_sandboxes. Drop it and substitute the local one.
        self.sandbox_client.teardown(wait=False)
        self.sandbox_client = LocalDockerSandboxClient()

    # ------------------------------------------------------------------ teardown
    def teardown_sandboxes(self) -> None:
        """Override SandboxMixin.teardown_sandboxes to stay off prime_sandboxes.

        Container kill+remove is handled by ``self.sandbox_client.teardown()``
        (called via ``teardown_mixin_sandbox_client``); here we just clear the
        active-set bookkeeping so the upstream handler doesn't think there's
        leftover state.
        """
        self.active_sandboxes.clear()

    async def teardown(self) -> None:
        """Override RLMExecutor.teardown to drop the inline prime SandboxClient.

        Mirrors the upstream method (rlm_env.py:1641-1666) minus the
        ``sync_client = SandboxClient(APIClient())`` block — registered sandbox
        ids are deleted via ``self.sandbox_client`` (the local one) instead.
        """
        if self._sessions:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            for session in sessions:
                try:
                    await self._stop_worker(session)
                finally:
                    if session.sandbox_id:
                        self.active_sandboxes.add(session.sandbox_id)
                    if session.local_rollout_dir not in self._retained_dirs:
                        shutil.rmtree(session.local_rollout_dir, True)
        if self.active_sandboxes:
            for sid in list(self.active_sandboxes):
                try:
                    await self.sandbox_client.delete(sid)
                finally:
                    self.deregister_sandbox(sid)
        self.teardown_sandbox_client()
