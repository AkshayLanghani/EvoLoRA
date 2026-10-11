"""Voice teardown must finish even when optional audio/network components stall."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from evolora.voice.narrator import Narrator


@pytest.fixture
def dictation_module(monkeypatch):
    # Import the implementation with stubbed optional dependencies, never audio hardware.
    stt = ModuleType("livekit.agents.stt")
    stt.STT = object
    stt.SpeechEventType = SimpleNamespace(FINAL_TRANSCRIPT=1, INTERIM_TRANSCRIPT=2)
    audio = ModuleType("evolora.voice.audio_io")
    audio.MicStream = object
    monkeypatch.setitem(sys.modules, "livekit.agents.stt", stt)
    monkeypatch.setitem(sys.modules, "evolora.voice.audio_io", audio)
    spec = importlib.util.spec_from_file_location(
        "evolora.voice._shutdown_test", Path("src/evolora/voice/dictation.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._PUMP_STOP_TIMEOUT = 0.01
    module._STT_STOP_TIMEOUT = 0.01
    return module


async def test_dictation_stalled_pump_is_cancelled_and_stream_closed(dictation_module):
    mic = Mock()
    session = dictation_module.DictationSession(
        Mock(), mic, asyncio.get_running_loop(), lambda text: None
    )
    session._finals = ["invoice details"]
    session._pump_task = asyncio.create_task(asyncio.Event().wait())
    session._read_task = asyncio.create_task(asyncio.Event().wait())
    stream = SimpleNamespace(aclose=AsyncMock())
    session._stt_stream = stream
    assert await asyncio.wait_for(session.stop(), timeout=0.5) == "invoice details"
    assert session._pump_task.cancelled()
    assert session._read_task.cancelled()
    stream.aclose.assert_awaited_once()
    mic.stop.assert_called_once()


async def test_dictation_cancelled_tasks_do_not_skip_cleanup(dictation_module):
    mic = Mock()
    mic.stop.side_effect = RuntimeError("device disappeared")
    session = dictation_module.DictationSession(
        Mock(), mic, asyncio.get_running_loop(), lambda text: None
    )
    task = asyncio.create_task(asyncio.Event().wait())
    task.cancel()
    await asyncio.sleep(0)
    session._pump_task = task
    session._read_task = task
    session._stt_stream = SimpleNamespace(aclose=AsyncMock())
    assert await session.stop() == ""
    session._stt_stream.aclose.assert_awaited_once()


async def test_dictation_stalled_stream_close_is_bounded(dictation_module):
    session = dictation_module.DictationSession(
        Mock(), Mock(), asyncio.get_running_loop(), lambda text: None
    )
    session._stt_stream = SimpleNamespace(aclose=AsyncMock(side_effect=asyncio.Event().wait))
    assert await asyncio.wait_for(session.stop(), timeout=0.5) == ""


async def test_narrator_already_cancelled_worker_closes_without_raising():
    narrator = Narrator(
        speak=AsyncMock(), muted=lambda: False, loop=asyncio.get_running_loop()
    )
    narrator.start()
    narrator._task.cancel()
    await asyncio.sleep(0)
    await narrator.aclose()
    assert narrator._task is None
