"""Tests for the on-device training backend and model runner (with a fake worker)."""

import json
import sys
from pathlib import Path

import pytest

from evolora.models.core import AgentPlan, LoraHyperparams, TrainingDataSpec
from evolora.training import local_backend
from evolora.training.local_backend import (
    LocalModelRunner,
    LocalTrainingBackend,
    WorkerError,
)

FAKE_WORKER = Path(__file__).parent / "fixtures" / "fake_worker_python.py"


@pytest.fixture(autouse=True)
def fake_worker(monkeypatch):
    monkeypatch.setattr(local_backend, "WORKER", FAKE_WORKER)
    monkeypatch.delenv("FAKE_WORKER_MODE", raising=False)


def _plan(batch_size: int = 4) -> AgentPlan:
    return AgentPlan(
        hyperparams=LoraHyperparams(
            r=16, lora_alpha=32, learning_rate=2e-4, num_epochs=2, batch_size=batch_size
        ),
        data_spec=TrainingDataSpec(
            examples=[{"prompt": f"Return JSON for {i}", "completion": f'{{"x": {i}}}'} for i in range(3)],
            rationale="test",
            max_examples=3,
        ),
        rationale="test plan",
        focus_areas=["json_format"],
    )


def _backend(tmp_path, **kwargs) -> LocalTrainingBackend:
    return LocalTrainingBackend(
        python=sys.executable,
        workspace=tmp_path / "workspace",
        adapters_dir=tmp_path / "adapters",
        **kwargs,
    )


async def _collect(backend, plan, goal="Make a model that specializes in strict JSON"):
    stream = await backend.train("run12345678", 1, plan, "Qwen/Qwen2.5-0.5B-Instruct", {"goal": goal})
    return [event async for event in stream]


async def test_training_streams_progress_and_archives_adapter(tmp_path):
    events = await _collect(_backend(tmp_path), _plan())

    progress = [e for e in events if "step" in e]
    assert [e["step"] for e in progress] == [1, 2, 3]
    assert progress[0]["total_steps"] == 3 and progress[0]["loss"] == 2.0
    assert any("Intel(R) Arc(TM) 140V" in e.get("message", "") for e in events)

    artifact = events[-1]["artifact"]
    archive = Path(artifact.adapter_path)
    assert archive.parent == tmp_path / "adapters"
    assert archive.name == "strict-json-run123-it1"
    assert (archive / "adapter_model.safetensors").is_file()
    assert artifact.is_mock is False and len(artifact.checksum) == 64

    meta = json.loads((archive / "evolora.json").read_text())
    assert meta["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert meta["goal"] == "Make a model that specializes in strict JSON"
    assert meta["examples"] == 3


async def test_batch_is_capped_with_gradient_accumulation(tmp_path):
    await _collect(_backend(tmp_path, max_batch=2), _plan(batch_size=4))
    job = json.loads((tmp_path / "workspace/run12345678/iter-1/train-job.json").read_text())
    assert job["hyperparams"]["batch_size"] == 2
    assert job["hyperparams"]["grad_accum"] == 2  # effective batch stays 4


async def test_out_of_memory_retries_once_smaller(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_WORKER_MODE", "oom_once")
    events = await _collect(_backend(tmp_path, max_length=512), _plan())

    assert any("Out of memory" in e.get("message", "") for e in events)
    job = json.loads((tmp_path / "workspace/run12345678/iter-1/train-job.json").read_text())
    assert job["hyperparams"]["batch_size"] == 1
    assert job["max_length"] == 256
    assert events[-1]["done"] is True


async def test_worker_failure_raises_with_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_WORKER_MODE", "fail")
    with pytest.raises(WorkerError, match="boom"):
        await _collect(_backend(tmp_path), _plan())


async def test_runner_answers_prompts_with_and_without_adapter(tmp_path):
    runner = LocalModelRunner(
        python=sys.executable, workspace=tmp_path / "ws", base_model_id="Qwen/Qwen2.5-0.5B-Instruct"
    )
    prompts = [{"sample_id": "a", "prompt": "first prompt"}, {"sample_id": "b", "prompt": "second"}]

    base = await runner.run_batch(prompts)
    tuned = await runner.run_batch(prompts, adapter_path=str(tmp_path / "adapter"))

    assert set(base) == {"a", "b"}
    assert not base["a"].endswith("(adapter)")
    assert tuned["a"].endswith("(adapter)")


async def test_relative_paths_work(tmp_path, monkeypatch):
    # The config defaults are relative (./workspace, ./adapters) while the worker runs
    # inside the job directory.
    monkeypatch.chdir(tmp_path)
    backend = LocalTrainingBackend(python=sys.executable, workspace="workspace", adapters_dir="adapters")
    events = await _collect(backend, _plan())
    assert Path(events[-1]["artifact"].adapter_path).is_dir()

    runner = LocalModelRunner(python=sys.executable, workspace="workspace", base_model_id="m")
    assert await runner.run_batch([{"sample_id": "a", "prompt": "p"}]) == {"a": "answer to p"}


async def test_health_check_needs_the_interpreter(tmp_path):
    assert await _backend(tmp_path).health_check()
    missing = LocalTrainingBackend(
        python=str(tmp_path / "nope.exe"), workspace=tmp_path, adapters_dir=tmp_path
    )
    assert not await missing.health_check()
