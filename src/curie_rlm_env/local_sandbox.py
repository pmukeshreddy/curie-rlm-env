"""Local Docker-backed sandbox client for CurieRLMEnv.

Implements the small subset of the prime_sandboxes.AsyncSandboxClient interface
that verifiers' SandboxMixin actually calls during a rollout — `create`,
`wait_for_creation`, `delete`, `bulk_delete`, `execute_command`, `upload_file`,
`download_file`, `read_file`, `run_background_job`, and `teardown`. Every
"sandbox" is a long-lived Docker container on the local daemon, so a single
GPU pod can run rollouts without depending on Prime's hosted sandbox API
(which raises `SandboxCreationError('Failed to create sandbox: No API key
configured. Set PRIME_API_KEY environment variable.')` from
prime_sandboxes/core/client.py).

Trust model: this client is intended for a single-tenant trusted GPU pod where
the trainer process IS the trust boundary. The container provides per-rollout
filesystem isolation, memory/CPU caps, and a process namespace, but it does
not protect against malicious code escaping a Docker container with the daemon
socket exposed. Do not use on shared infrastructure.

Optional dependency: the `docker` Python SDK is lazy-imported. Construction
and configuration parsing work without it; only methods that actually touch
the daemon import it.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import shlex
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from prime_sandboxes import (
    APIError,
    CommandTimeoutError,
    SandboxFileNotFoundError,
    SandboxOOMError,
    SandboxTimeoutError,
    UploadTimeoutError,
)

logger = logging.getLogger(__name__)


@dataclass
class _SandboxRef:
    """Minimal stand-in for prime_sandboxes.models.Sandbox; SandboxMixin only reads `.id`."""

    id: str


@dataclass
class _CommandResponse:
    """Stand-in for prime_sandboxes.models.CommandResponse (`.exit_code`, `.stdout`, `.stderr`)."""

    exit_code: int
    stdout: str
    stderr: str


@dataclass
class _ReadFileResponse:
    """Stand-in for prime_sandboxes.models.ReadFileResponse (`.content`)."""

    content: str


@dataclass
class _BulkDeleteResponse:
    deleted: list[str]


def _import_docker() -> Any:
    """Lazy-import the docker SDK with an actionable error if missing/unreachable."""
    try:
        import docker  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover — environment-dependent
        raise RuntimeError(
            "LocalDockerSandboxClient requires the `docker` Python package. "
            "Install with `uv pip install docker`, or set "
            "CURIE_SANDBOX_BACKEND=prime to use the hosted sandbox (which "
            "requires PRIME_API_KEY)."
        ) from exc
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # pragma: no cover — environment-dependent
        raise RuntimeError(
            f"Local Docker daemon unreachable ({type(exc).__name__}: {exc}). "
            "Start the Docker daemon, or set CURIE_SANDBOX_BACKEND=prime to use "
            "the hosted sandbox (which requires PRIME_API_KEY)."
        ) from exc
    return client


def _seconds(value: Any, default: float = 60.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class LocalDockerSandboxClient:
    """Drop-in replacement for prime_sandboxes.AsyncSandboxClient.

    Each call dispatches to the local Docker daemon. State is the dict of
    `{sandbox_id: container}`. The class is async-only because that's what
    SandboxMixin awaits.
    """

    def __init__(self, **_kwargs: Any) -> None:
        # _kwargs accepted for compat with ThreadedAsyncSandboxClient(max_connections=...).
        self._docker: Optional[Any] = None
        self._containers: dict[str, Any] = {}

    # ------------------------------------------------------------------ docker
    @property
    def _client(self) -> Any:
        if self._docker is None:
            self._docker = _import_docker()
        return self._docker

    def _container(self, sandbox_id: str) -> Any:
        c = self._containers.get(sandbox_id)
        if c is None:
            raise APIError(f"Local sandbox {sandbox_id} not found")
        return c

    # ------------------------------------------------------------------ create
    async def create(self, request: Any) -> _SandboxRef:
        # `request` is a prime_sandboxes.CreateSandboxRequest pydantic model.
        sandbox_id = f"curie-local-{uuid.uuid4().hex[:12]}"
        image = getattr(request, "docker_image", None) or "python:3.11-slim"
        memory_gb = float(getattr(request, "memory_gb", 2.0) or 2.0)
        cpu_cores = float(getattr(request, "cpu_cores", 1.0) or 1.0)
        start_command = getattr(request, "start_command", None) or "tail -f /dev/null"
        env = dict(getattr(request, "environment_vars", {}) or {})

        def _run() -> Any:
            return self._client.containers.run(
                image=image,
                command=["sh", "-c", start_command],
                detach=True,
                name=sandbox_id,
                mem_limit=f"{int(memory_gb * 1024)}m",
                nano_cpus=int(cpu_cores * 1_000_000_000),
                environment=env,
                labels={"curie_rlm_env": "1", "sandbox_id": sandbox_id},
                network_mode=os.environ.get("CURIE_SANDBOX_NETWORK", "bridge"),
                auto_remove=False,
            )

        try:
            container = await asyncio.to_thread(_run)
        except Exception as exc:
            raise APIError(f"Local Docker create failed: {exc}") from exc
        self._containers[sandbox_id] = container
        return _SandboxRef(id=sandbox_id)

    async def wait_for_creation(self, sandbox_id: str, max_attempts: int = 120) -> _SandboxRef:
        # Containers go to "running" essentially immediately after `containers.run`.
        # We still poll briefly to cover slow image pulls and avoid a race against
        # the very first exec_run.
        container = self._container(sandbox_id)
        deadline = time.monotonic() + max(1, max_attempts)  # max_attempts ≈ seconds budget

        def _refresh_status() -> str:
            container.reload()
            return container.status

        while True:
            status = await asyncio.to_thread(_refresh_status)
            if status == "running":
                return _SandboxRef(id=sandbox_id)
            if status in {"exited", "dead", "removing"}:
                raise APIError(
                    f"Local sandbox {sandbox_id} failed to start (status={status})"
                )
            if time.monotonic() > deadline:
                raise APIError(
                    f"Local sandbox {sandbox_id} not ready within {max_attempts}s"
                )
            await asyncio.sleep(0.25)

    # ------------------------------------------------------------------ delete
    async def delete(self, sandbox_id: str) -> dict:
        container = self._containers.pop(sandbox_id, None)
        if container is None:
            return {"id": sandbox_id, "status": "not_found"}

        def _kill_and_remove() -> None:
            try:
                container.kill()
            except Exception:
                pass
            try:
                container.remove(force=True)
            except Exception:
                pass

        await asyncio.to_thread(_kill_and_remove)
        return {"id": sandbox_id, "status": "deleted"}

    async def bulk_delete(self, sandbox_ids: list[str]) -> _BulkDeleteResponse:
        deleted: list[str] = []
        for sid in sandbox_ids:
            await self.delete(sid)
            deleted.append(sid)
        return _BulkDeleteResponse(deleted=deleted)

    # ------------------------------------------------------------------ exec
    async def execute_command(
        self,
        sandbox_id: str,
        command: str,
        timeout: Optional[float] = None,
        working_dir: Optional[str] = None,
        **_kwargs: Any,
    ) -> _CommandResponse:
        container = self._container(sandbox_id)
        wrapped = command
        if working_dir:
            wrapped = f"cd {shlex.quote(working_dir)} && {command}"
        # Docker exec doesn't enforce timeout natively; wrap with `timeout` shell util.
        eff_timeout = _seconds(timeout, default=120.0)
        wrapped = f"timeout --preserve-status {int(eff_timeout)}s sh -c {shlex.quote(wrapped)}"

        def _exec() -> tuple[int, bytes, bytes]:
            res = container.exec_run(
                cmd=["sh", "-c", wrapped],
                demux=True,  # split stdout/stderr
                tty=False,
            )
            stdout_b, stderr_b = res.output if isinstance(res.output, tuple) else (res.output, b"")
            return res.exit_code, stdout_b or b"", stderr_b or b""

        try:
            exit_code, stdout_b, stderr_b = await asyncio.to_thread(_exec)
        except Exception as exc:
            raise APIError(f"Local exec failed in {sandbox_id}: {exc}") from exc

        # `timeout` returns 124 when it killed the child. Surface as CommandTimeoutError so
        # SandboxMixin's existing retry/error handling matches the hosted backend.
        if exit_code == 124:
            raise CommandTimeoutError(
                f"Command in {sandbox_id} exceeded {int(eff_timeout)}s timeout"
            )
        return _CommandResponse(
            exit_code=int(exit_code or 0),
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
        )

    # ------------------------------------------------------------------ upload
    async def upload_file(
        self,
        sandbox_id: str,
        remote_path: str,
        local_path: str,
    ) -> dict:
        container = self._container(sandbox_id)
        local = Path(local_path)
        if not local.exists():
            raise FileNotFoundError(f"Local source missing: {local_path}")

        def _put() -> None:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                info = tarfile.TarInfo(name=Path(remote_path).name)
                data = local.read_bytes()
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            buf.seek(0)
            target_dir = str(Path(remote_path).parent) or "/"
            container.exec_run(["mkdir", "-p", target_dir])
            container.put_archive(target_dir, buf.getvalue())

        try:
            await asyncio.to_thread(_put)
        except Exception as exc:
            raise UploadTimeoutError(
                f"Local upload of {local_path} → {sandbox_id}:{remote_path} failed: {exc}"
            ) from exc
        return {"path": remote_path, "size": local.stat().st_size}

    async def download_file(
        self,
        sandbox_id: str,
        remote_path: str,
        local_path: str,
    ) -> None:
        container = self._container(sandbox_id)

        def _get() -> None:
            try:
                stream, _stat = container.get_archive(remote_path)
            except Exception as exc:
                raise SandboxFileNotFoundError(
                    f"Remote path {remote_path} not found in {sandbox_id}"
                ) from exc
            buf = io.BytesIO(b"".join(stream))
            buf.seek(0)
            with tarfile.open(fileobj=buf, mode="r") as tar:
                member_name = Path(remote_path).name
                member = tar.getmember(member_name)
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise SandboxFileNotFoundError(
                        f"Remote path {remote_path} not a regular file in {sandbox_id}"
                    )
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                Path(local_path).write_bytes(extracted.read())

        await asyncio.to_thread(_get)

    async def read_file(
        self,
        sandbox_id: str,
        remote_path: str,
        timeout: Optional[float] = None,
    ) -> _ReadFileResponse:
        container = self._container(sandbox_id)

        def _read() -> str:
            try:
                stream, _stat = container.get_archive(remote_path)
            except Exception as exc:
                raise SandboxFileNotFoundError(
                    f"Remote path {remote_path} not found in {sandbox_id}"
                ) from exc
            buf = io.BytesIO(b"".join(stream))
            buf.seek(0)
            with tarfile.open(fileobj=buf, mode="r") as tar:
                member_name = Path(remote_path).name
                member = tar.getmember(member_name)
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise SandboxFileNotFoundError(
                        f"Remote path {remote_path} not a regular file in {sandbox_id}"
                    )
                return extracted.read().decode("utf-8", errors="replace")

        content = await asyncio.to_thread(_read)
        return _ReadFileResponse(content=content)

    # ------------------------------------------------------------------ background
    async def run_background_job(
        self,
        sandbox_id: str,
        command: str,
        timeout: int,
        working_dir: Optional[str] = None,
        poll_interval: int = 3,
    ) -> _CommandResponse:
        # Local Docker doesn't have a dedicated background-job API; just exec
        # synchronously within the timeout and translate timeouts.
        try:
            return await self.execute_command(
                sandbox_id,
                command,
                timeout=timeout,
                working_dir=working_dir,
            )
        except CommandTimeoutError as exc:
            raise SandboxTimeoutError(
                f"Background job in {sandbox_id} exceeded {timeout}s"
            ) from exc

    # ------------------------------------------------------------------ teardown
    def teardown(self, wait: bool = True) -> None:
        """Kill+remove every tracked container; safe to call multiple times."""
        for sandbox_id in list(self._containers):
            container = self._containers.pop(sandbox_id, None)
            if container is None:
                continue
            try:
                container.kill()
            except Exception:
                pass
            try:
                container.remove(force=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_BACKEND_ENV = "CURIE_SANDBOX_BACKEND"
_BACKEND_LOCAL = "local_docker"
_BACKEND_PRIME = "prime"
_VALID_BACKENDS = {_BACKEND_LOCAL, _BACKEND_PRIME}


def resolve_sandbox_backend() -> str:
    """Return 'local_docker' (default) or 'prime'.

    Set CURIE_SANDBOX_BACKEND to override. Any value other than the two valid
    backends raises ValueError so misconfiguration is loud, not silent.
    """
    value = os.environ.get(_BACKEND_ENV, _BACKEND_LOCAL).strip().lower()
    if value not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid {_BACKEND_ENV}={value!r}; expected one of {sorted(_VALID_BACKENDS)}"
        )
    return value
