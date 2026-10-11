"""Worker failures must remain explicit rather than becoming zero-score artifacts."""

import json
import sys
from pathlib import Path

import pytest

from evolora.models.core import AgentPlan, LoraHyperparams
from evolora.training import local_backend
from evolora.training.local_backend import LocalModelRunner, LocalTrainingBackend, WorkerError


def backend(tmp_path, cap=3):
    return LocalTrainingBackend(
        python=sys.executable, workspace=tmp_path / "ws", adapters_dir=tmp_path / "adapters",
        max_batch=cap,
    )


@pytest.mark.parametrize("effective,cap", [(4, 3), (8, 5), (6, 4), (7, 3), (32, 31)])
def test_non_divisor_cap_preserves_effective_batch(tmp_path, effective, cap):
    hp = backend(tmp_path)._hyperparams(
        AgentPlan(hyperparams=LoraHyperparams(batch_size=effective)), cap,
    )
    assert hp["batch_size"] <= cap
    assert hp["batch_size"] * hp["grad_accum"] == effective


@pytest.mark.parametrize("mode", ["no_done", "no_weights", "empty_weights", "no_config", "outside"])
async def test_incomplete_training_never_archives(tmp_path, monkeypatch, mode):
    async def worker(python, command, job, job_dir):
        if mode == "no_done":
            return
        source = (tmp_path / "foreign") if mode == "outside" else job_dir / "adapter"
        source.mkdir(parents=True)
        if mode != "no_weights":
            (source / "adapter_model.safetensors").write_bytes(
                b"" if mode == "empty_weights" else b"weights",
            )
        if mode != "no_config":
            (source / "adapter_config.json").write_text("{}")
        yield {"event": "done", "adapter_dir": str(source)}

    monkeypatch.setattr(local_backend, "run_worker", worker)
    stream = await backend(tmp_path).train("run", 1, AgentPlan(), "model")
    with pytest.raises(WorkerError):
        _ = [event async for event in stream]
    assert not (tmp_path / "adapters").exists()


@pytest.mark.parametrize("responses", [None, [], {"b": "answer"}, {"a": 42}, {"a": "x", "b": "y"}])
async def test_malformed_generation_is_an_execution_failure(tmp_path, monkeypatch, responses):
    async def worker(python, command, job, job_dir):
        job_dir.mkdir(parents=True)
        if responses is not None:
            Path(job["output_file"]).write_text(json.dumps(responses))
        yield {"event": "done"}

    monkeypatch.setattr(local_backend, "run_worker", worker)
    runner = LocalModelRunner(python=sys.executable, workspace=tmp_path, base_model_id="m")
    with pytest.raises(WorkerError):
        await runner.run_batch([{"sample_id": "a", "prompt": "p"}])


async def test_duplicate_ids_rejected_before_worker(tmp_path, monkeypatch):
    async def worker(*args):
        raise AssertionError("worker must not start")
        yield

    monkeypatch.setattr(local_backend, "run_worker", worker)
    runner = LocalModelRunner(python=sys.executable, workspace=tmp_path, base_model_id="m")
    with pytest.raises(WorkerError, match="duplicate"):
        await runner.run_batch([{"sample_id": "a"}, {"sample_id": "a"}])
    assert await runner.run_batch([]) == {}


@pytest.mark.parametrize("cpu_fails", [False, True])
async def test_unstable_training_retries_cpu_once_preserving_data(tmp_path, monkeypatch, cpu_fails):
    jobs = []

    async def worker(python, command, job, job_dir):
        jobs.append(job)
        if len(jobs) == 1 or cpu_fails:
            raise WorkerError("non-finite gradient", unstable=True)
        source = job_dir / "adapter"
        source.mkdir(parents=True)
        (source / "adapter_model.safetensors").write_bytes(b"weights")
        (source / "adapter_config.json").write_text("{}")
        yield {"event": "done", "adapter_dir": str(source), "device": "cpu"}

    monkeypatch.setattr(local_backend, "run_worker", worker)
    plan = AgentPlan(hyperparams=LoraHyperparams(batch_size=4))
    stream = await backend(tmp_path).train("run", 1, plan, "model")
    if cpu_fails:
        with pytest.raises(WorkerError, match="non-finite"):
            _ = [event async for event in stream]
        assert not (tmp_path / "adapters").exists()
    else:
        events = [event async for event in stream]
        assert any("retrying once on CPU" in event.get("message", "") for event in events)
        meta = json.loads((Path(events[-1]["artifact"].adapter_path) / "evolora.json").read_text())
        assert meta["execution_device"] == "cpu"
    assert len(jobs) == 2
    assert jobs[1]["device"] == "cpu"
    assert jobs[0]["examples"] == jobs[1]["examples"]
    assert jobs[0]["max_length"] == jobs[1]["max_length"]
    assert jobs[1]["hyperparams"]["batch_size"] == 1
    assert jobs[1]["hyperparams"]["grad_accum"] == 4


async def test_closing_training_stream_waits_for_worker_cleanup(tmp_path, monkeypatch):
    closed = []

    async def worker(*args):
        try:
            yield {"event": "loaded", "device_name": "CPU"}
            raise AssertionError("cancelled worker resumed")
        finally:
            closed.append(True)

    monkeypatch.setattr(local_backend, "run_worker", worker)
    stream = await backend(tmp_path).train("run", 1, AgentPlan(), "model")
    await anext(stream)
    await stream.aclose()
    assert closed == [True]
