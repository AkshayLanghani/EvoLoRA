"""GPU admission and cleanup must cover the complete HTTP operation."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from evolora.config import Config
from evolora.web import app as web


@pytest.fixture
def local_api(monkeypatch):
    cfg = Config(profile="local")
    monkeypatch.setattr(web, "get_config", lambda: cfg)
    monkeypatch.setattr(web, "_active_runs", 0)
    monkeypatch.setattr(web, "_active_chat", False)
    monkeypatch.setattr(web, "local_training_issue", lambda cfg: None)
    monkeypatch.setattr(web, "build_resources", lambda *args, **kwargs: None)
    return cfg


def request():
    return Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1234)})


@pytest.mark.asyncio
async def test_run_reserves_before_streaming(local_api, monkeypatch):
    monkeypatch.setattr(web, "_build_orchestrator", lambda req: object())
    response = await web.start_run(web.RunRequest(), request())
    assert web._active_runs == 1
    with pytest.raises(HTTPException) as error:
        await web.start_run(web.RunRequest(), request())
    assert error.value.status_code == 503
    assert isinstance(response, web._RunResponse)


@pytest.mark.asyncio
async def test_response_releases_slot_when_delivery_fails(local_api, monkeypatch):
    monkeypatch.setattr(web, "_active_runs", 1)

    async def fail(*args):
        raise RuntimeError("disconnected before streaming")

    monkeypatch.setattr(web.StreamingResponse, "__call__", fail)
    response = web._RunResponse(iter(()))
    with pytest.raises(RuntimeError, match="disconnected"):
        await response({}, None, None)
    assert web._active_runs == 0


@pytest.mark.asyncio
async def test_chat_blocks_other_gpu_operations_and_releases(local_api, monkeypatch):
    started = asyncio.Event()
    finish = asyncio.Event()

    async def run_batch(*args):
        started.set()
        await finish.wait()
        return {"chat": "answer"}

    monkeypatch.setattr(web, "get_runner", lambda *args, **kw: SimpleNamespace(run_batch=run_batch))
    task = asyncio.create_task(web.chat(web.ChatRequest(prompt="first")))
    await started.wait()
    try:
        with pytest.raises(HTTPException) as chat_error:
            await web.chat(web.ChatRequest(prompt="second"))
        assert chat_error.value.status_code == 503
        with pytest.raises(HTTPException) as run_error:
            await web.start_run(web.RunRequest(), request())
        assert run_error.value.status_code == 503
    finally:
        finish.set()
        result = await task
    assert result["answer"] == "answer"
    assert web._active_chat is False


@pytest.mark.asyncio
async def test_cancelled_chat_releases_gpu_slot(local_api, monkeypatch):
    started = asyncio.Event()

    async def run_batch(*args):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(web, "get_runner", lambda *args, **kw: SimpleNamespace(run_batch=run_batch))
    task = asyncio.create_task(web.chat(web.ChatRequest(prompt="first")))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert web._active_chat is False


@pytest.mark.asyncio
async def test_chat_resource_failure_releases_slot(local_api, monkeypatch):
    async def acquire(resource):
        raise RuntimeError("device unavailable")

    monkeypatch.setattr(
        web, "build_resources", lambda *args, **kw: SimpleNamespace(acquire=acquire)
    )
    with pytest.raises(HTTPException) as error:
        await web.chat(web.ChatRequest(prompt="first"))
    assert error.value.status_code == 500
    assert web._active_chat is False


@pytest.mark.asyncio
async def test_stream_disconnect_explicitly_closes_orchestrator(local_api):
    from evolora.models.events import Event, EventKind

    closed = False
    cancelled = False

    async def events():
        nonlocal closed
        try:
            yield Event(kind=EventKind.RUN_STARTED, run_id="disconnect-test")
            await asyncio.Event().wait()
        finally:
            closed = True

    async def run():
        return events()

    def cancel():
        nonlocal cancelled
        cancelled = True

    stream = web._stream_run(SimpleNamespace(run=run, cancel=cancel))
    await anext(stream)
    assert "disconnect-test" in web._live_runs
    await stream.aclose()
    assert closed and cancelled
    assert "disconnect-test" not in web._live_runs
