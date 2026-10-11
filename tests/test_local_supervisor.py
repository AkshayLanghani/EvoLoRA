"""Tests for the llama-server supervisor and the local resource manager."""

import socket
import sys
from pathlib import Path

import pytest

from evolora.local.supervisor import LlamaServer, LocalResources, ModelSpec

FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_llama_server.py"


class FakeLlamaServer(LlamaServer):
    """Runs the Python stand-in instead of a llama.cpp binary."""

    def command(self, model: ModelSpec, ctx: int | None = None) -> list[str]:
        return [sys.executable, self._server_path, *super().command(model, ctx)[1:]]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def models(tmp_path):
    planner = tmp_path / "planner.gguf"
    judge = tmp_path / "judge.gguf"
    planner.write_bytes(b"gguf")
    judge.write_bytes(b"gguf")
    return ModelSpec(str(planner), "planner-model"), ModelSpec(str(judge), "judge-model")


@pytest.fixture
def server(tmp_path):
    srv = FakeLlamaServer(
        server_path=str(FAKE_SERVER),
        port=_free_port(),
        log_path=tmp_path / "llama-server.log",
        startup_timeout_s=30,
    )
    yield srv
    srv._kill_now()


def test_command_binds_loopback_and_enables_tool_templates():
    srv = LlamaServer(server_path="llama-server", port=9000, extra_args="--threads 6")
    cmd = srv.command(ModelSpec("m.gguf", "m"))
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert "--jinja" in cmd
    assert cmd[-2:] == ["--threads", "6"]


async def test_start_swap_and_stop(server, models):
    planner, judge = models
    await server.start(planner)
    assert server.running
    assert server.loaded_model == planner
    assert await server.healthy()

    first_pid = server._proc.pid
    await server.start(planner)  # already loaded: no restart
    assert server._proc.pid == first_pid

    await server.start(judge)  # swap restarts with the other model
    assert server.loaded_model == judge
    assert server._proc.pid != first_pid

    await server.stop()
    assert not server.running
    assert not await server.healthy()


async def test_failed_start_is_retried(server, models, monkeypatch, tmp_path):
    # e.g. the GPU driver hasn't released memory yet right after a training job.
    monkeypatch.setattr("evolora.local.supervisor.RETRY_DELAY_S", 0.1)
    monkeypatch.setenv("FAKE_LLAMA_FAIL_ONCE_FILE", str(tmp_path / "failed-once"))
    planner, _ = models
    await server.start(planner)
    assert server.running and (tmp_path / "failed-once").exists()


def test_last_attempt_uses_a_smaller_context():
    srv = LlamaServer(server_path="llama-server", port=9000, ctx=8192)
    cmd = srv.command(ModelSpec("m.gguf", "m"), ctx=4096)
    assert cmd[cmd.index("--ctx-size") + 1] == "4096"


async def test_missing_binary_or_model_is_reported(tmp_path, models):
    planner, _ = models
    resources = LocalResources(
        base_url="http://127.0.0.1:1/v1",
        planner=planner,
        server=LlamaServer(server_path=str(tmp_path / "missing.exe"), port=_free_port()),
    )
    warning = await resources.acquire("planner")
    assert "llama-server not found" in warning
    assert await resources.acquire("planner") is None  # warned once per run


async def test_gpu_stage_releases_llm_memory(server, models):
    planner, judge = models
    resources = LocalResources(
        base_url=server.base_url + "/v1",
        planner=planner,
        judge=judge,
        server=server,
        release_for_gpu=True,
    )
    assert await resources.acquire("planner") is None
    assert server.loaded_model == planner
    assert await resources.acquire("judge") is None
    assert server.loaded_model == judge

    await resources.acquire("gpu")
    assert not server.running


async def test_mock_training_keeps_llm_loaded(server, models):
    planner, _ = models
    resources = LocalResources(base_url=server.base_url + "/v1", planner=planner, server=server)
    await resources.acquire("planner")
    await resources.acquire("gpu")
    assert server.running


async def test_unmanaged_mode_checks_reachability(server, models):
    planner, _ = models
    unmanaged = LocalResources(base_url=server.base_url + "/v1", planner=planner)
    assert "not reachable" in await unmanaged.acquire("planner")

    await server.start(planner)  # someone else runs the server
    assert await LocalResources(base_url=server.base_url + "/v1", planner=planner).acquire(
        "planner"
    ) is None
