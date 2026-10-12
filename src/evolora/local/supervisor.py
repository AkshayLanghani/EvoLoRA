"""Local LLM server lifecycle and memory handoff for the local profile.

A 16 GB laptop can't hold the planner LLM and a training run at the same time, so one
``llama-server`` process is started, swapped between the planner and judge models, and
stopped before GPU work. A stopped process returns all of its memory to the OS, which is
more dependable than unloading inside a long-lived Python process.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import shlex
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from evolora.orchestration.orchestrator import Resource

HEALTH_POLL_S = 0.5
STOP_GRACE_S = 10.0
START_ATTEMPTS = 3
RETRY_DELAY_S = 5.0


class LocalLLMError(RuntimeError):
    """The local LLM server could not be started or reached."""


@dataclass(frozen=True)
class ModelSpec:
    path: str
    alias: str


class LlamaServer:
    """Starts, swaps and stops one llama.cpp ``llama-server`` process on loopback."""

    def __init__(
        self,
        *,
        server_path: str,
        port: int,
        ctx: int = 16384,
        gpu_layers: int = 99,
        extra_args: str = "",
        log_path: str | Path = "logs/llama-server.log",
        startup_timeout_s: float = 300.0,
    ) -> None:
        self._server_path = server_path
        self._port = port
        self._ctx = ctx
        self._gpu_layers = gpu_layers
        self._extra_args = shlex.split(extra_args, posix=os.name != "nt") if extra_args else []
        self._log_path = Path(log_path)
        self._startup_timeout_s = startup_timeout_s
        self._proc: subprocess.Popen | None = None
        self._model: ModelSpec | None = None
        self._lock = asyncio.Lock()
        atexit.register(self._kill_now)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def loaded_model(self) -> ModelSpec | None:
        return self._model if self.running else None

    def command(self, model: ModelSpec, ctx: int | None = None) -> list[str]:
        return [
            self._server_path,
            "--model", model.path,
            "--alias", model.alias,
            "--host", "127.0.0.1",  # never expose the server beyond this machine
            "--port", str(self._port),
            "--ctx-size", str(ctx or self._ctx),
            "--n-gpu-layers", str(self._gpu_layers),
            "--jinja",  # chat templates with tool calling
            *self._extra_args,
        ]

    async def start(self, model: ModelSpec) -> None:
        """Ensure the server is up with ``model`` loaded, restarting it to swap models.

        Right after a training job exits, the GPU driver can take a few seconds to hand
        shared memory back, so a start that fails is retried after a pause, and the last
        attempt uses half the context window (the KV cache is the part that grows).
        """
        async with self._lock:
            if self.running and self._model == model:
                return
            await self._stop_locked()
            if not self._server_path or not Path(self._server_path).is_file():
                raise LocalLLMError(
                    f"llama-server not found at {self._server_path!r} (set LLAMA_SERVER_PATH)"
                )
            if not model.path or not Path(model.path).is_file():
                raise LocalLLMError(
                    f"model file not found at {model.path!r} (run `evolora setup` or set "
                    "LOCAL_LLM_MODEL_PATH)"
                )
            await asyncio.to_thread(self._free_port)
            for attempt in range(1, START_ATTEMPTS + 1):
                ctx = self._ctx if attempt < START_ATTEMPTS else max(2048, self._ctx // 2)
                try:
                    self._spawn(model, ctx)
                    await self._wait_healthy()
                    return
                except LocalLLMError:
                    if attempt == START_ATTEMPTS:
                        raise
                    await asyncio.sleep(RETRY_DELAY_S * attempt)

    def _spawn(self, model: ModelSpec, ctx: int) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self._log_path.open("ab")
        try:
            self._proc = subprocess.Popen(
                self.command(model, ctx),
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        finally:
            log.close()  # the child keeps its own handle
        self._model = model

    def _free_port(self) -> None:
        """Stop a leftover llama-server on our port (e.g. after the agent was killed)."""
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self._port)) != 0:
                return  # nothing listening
        if os.name != "nt":
            subprocess.run(
                ["pkill", "-f", f"llama-server.*--port {self._port}( |$)"],
                capture_output=True, timeout=30,
            )
            return
        script = (
            "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | "
            f"Where-Object {{ $_.CommandLine -match '--port {self._port}( |$)' }} | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def status(self) -> dict:
        return {
            "running": self.running,
            "model": self._model.alias if self.running and self._model else None,
            "port": self._port,
            "server_path": self._server_path,
            "log": str(self._log_path),
        }

    async def _wait_healthy(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._startup_timeout_s
        while loop.time() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                code = self._proc.returncode if self._proc else None
                self._proc = None
                raise LocalLLMError(
                    f"llama-server exited during startup (code {code}); see {self._log_path}"
                )
            if await self.healthy():  # 503 while the model is still loading
                return
            await asyncio.sleep(HEALTH_POLL_S)
        await self._stop_locked()
        raise LocalLLMError(f"llama-server not healthy after {self._startup_timeout_s:.0f}s")

    async def _stop_locked(self) -> None:
        proc, self._proc, self._model = self._proc, None, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            await asyncio.to_thread(proc.wait, STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            await asyncio.to_thread(proc.wait)

    def _kill_now(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()


class LocalResources:
    """ResourceManager for the local profile: one heavy model in memory at a time.

    Managed mode starts/swaps/stops llama-server itself. Unmanaged mode only checks that
    a server you started (llama-server, LM Studio, ...) is reachable. Failures never stop
    a run: they come back as warnings and the heuristic fallbacks take over.
    """

    def __init__(
        self,
        *,
        base_url: str,
        planner: ModelSpec,
        judge: ModelSpec | None = None,
        server: LlamaServer | None = None,
        release_for_gpu: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._planner = planner
        self._judge = judge or planner
        self._server = server
        self._release_for_gpu = release_for_gpu
        self._warned: set[str] = set()

    @property
    def managed(self) -> bool:
        return self._server is not None

    async def acquire(self, resource: Resource) -> str | None:
        if resource == "gpu":
            if self._server is not None and self._release_for_gpu and self._server.running:
                await self._server.stop()
            return None

        model = self._judge if resource == "judge" else self._planner
        if self._server is None:
            if await self._reachable():
                return None
            return self._warn_once(
                f"Local LLM not reachable at {self._base_url}; heuristic fallbacks will be used"
            )
        try:
            await self._server.start(model)
        except LocalLLMError as exc:
            return self._warn_once(f"Local LLM unavailable: {exc}. Heuristic fallbacks will be used")
        return None

    async def shutdown(self) -> None:
        if self._server is not None:
            await self._server.stop()

    def status(self) -> dict:
        info: dict = {
            "managed": self.managed,
            "base_url": self._base_url,
            "planner_model": self._planner.alias,
            "judge_model": self._judge.alias,
        }
        if self._server is not None:
            info["server"] = self._server.status()
        return info

    async def _reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._base_url}/models")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def _warn_once(self, message: str) -> str | None:
        if message in self._warned:
            return None
        self._warned.add(message)
        return message
