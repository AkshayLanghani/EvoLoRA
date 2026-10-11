"""On-device training backend and model runner for the local profile.

Both launch ``evolora/local/worker.py`` with the training interpreter, one job per
process: training writes a LoRA adapter, generation answers eval prompts. Scoring stays
in the main process against the locked expected answers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import subprocess
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

from evolora.local.installer import hf_env
from evolora.models.core import AgentPlan, ArtifactMeta
from evolora.training.backends import _adapter_label

WORKER = Path(__file__).resolve().parents[1] / "local" / "worker.py"
MARKER = "EVOLORA "
TAIL_LINES = 30


class WorkerError(RuntimeError):
    def __init__(self, message: str, *, oom: bool = False) -> None:
        super().__init__(message)
        self.oom = oom


async def run_worker(python: str, command: str, job: dict, job_dir: Path) -> AsyncIterator[dict]:
    """Run one worker job and yield its ``EVOLORA`` events.

    Raises WorkerError when the job fails. Closing the iterator early (cancellation)
    kills the process so the GPU is freed immediately.
    """
    # The worker runs with job_dir as its working directory: every path must be absolute.
    job_dir = job_dir.resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    job_file = job_dir / f"{command}-job.json"
    job_file.write_text(json.dumps(job), encoding="utf-8")
    log_file = job_dir / f"{command}.log"

    proc = subprocess.Popen(
        [python, "-u", str(WORKER), command, str(job_file)],
        cwd=job_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=hf_env(),  # the worker may download the base model on first use
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def pump() -> None:
        with log_file.open("w", encoding="utf-8") as log:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                log.write(line + "\n")
                loop.call_soon_threadsafe(queue.put_nowait, line)
        loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=pump, daemon=True).start()
    tail: list[str] = []
    error: dict | None = None
    try:
        while (line := await queue.get()) is not None:
            # Progress bars can share a line with an event, so look anywhere in it.
            at = line.find(MARKER)
            if at >= 0:
                event = json.loads(line[at + len(MARKER):])
                if event.get("event") == "error":
                    error = event
                else:
                    yield event
            else:
                tail = (tail + [line])[-TAIL_LINES:]
        code = await asyncio.to_thread(proc.wait)
    finally:
        if proc.poll() is None:
            proc.kill()
    if code != 0 or error:
        detail = error["error"] if error else "\n".join(tail[-5:]) or f"exit code {code}"
        raise WorkerError(f"{command} failed: {detail} (log: {log_file})",
                          oom=bool(error and error.get("oom")))


def _adapter_checksum(adapter_dir: Path) -> str:
    digest = hashlib.sha256()
    for f in sorted(adapter_dir.glob("adapter_model.*")):
        digest.update(f.read_bytes())
    return digest.hexdigest()


class LocalTrainingBackend:
    """Trains a LoRA adapter on this machine (Intel XPU, CUDA or CPU, picked by the worker)."""

    is_mock = False
    name = "local"

    def __init__(
        self,
        *,
        python: str,
        workspace: str | Path,
        adapters_dir: str | Path,
        max_length: int = 512,
        max_batch: int = 2,
    ) -> None:
        self._python = python
        self._workspace = Path(workspace).resolve()
        self._adapters = Path(adapters_dir).resolve()
        self._max_length = max_length
        self._max_batch = max(1, max_batch)

    async def health_check(self) -> bool:
        return Path(self._python).is_file()

    async def train(
        self,
        run_id: str,
        iteration: int,
        plan: AgentPlan,
        base_model_id: str,
        remote_payload: dict | None = None,
    ) -> AsyncIterator[dict]:
        return self._stream(run_id, iteration, plan, base_model_id, remote_payload or {})

    def _hyperparams(self, plan: AgentPlan, batch_cap: int) -> dict:
        hp = plan.hyperparams
        # Keep the agent's effective batch size while capping memory per step.
        per_device = min(hp.batch_size, batch_cap)
        return {
            "r": hp.r,
            "lora_alpha": hp.lora_alpha,
            "learning_rate": hp.learning_rate,
            "num_epochs": hp.num_epochs,
            "batch_size": per_device,
            "grad_accum": math.ceil(hp.batch_size / per_device),
        }

    async def _stream(
        self, run_id: str, iteration: int, plan: AgentPlan, base_model_id: str, payload: dict
    ) -> AsyncIterator[dict]:
        job_dir = self._workspace / run_id / f"iter-{iteration}"
        started = time.perf_counter()
        batch_cap, max_length = self._max_batch, self._max_length
        done: dict | None = None

        for attempt in (1, 2):
            job = {
                "base_model": base_model_id,
                "examples": plan.data_spec.examples,
                "hyperparams": self._hyperparams(plan, batch_cap),
                "max_length": max_length,
                "gradient_checkpointing": True,
                "output_dir": str(job_dir),
            }
            try:
                async for event in run_worker(self._python, "train", job, job_dir):
                    if event["event"] == "progress":
                        yield {k: event[k] for k in ("step", "total_steps", "loss")} | {"done": False}
                    elif event["event"] == "loaded":
                        yield {"phase": "train", "done": False,
                               "message": f"Training {base_model_id} on {event['device_name']}"}
                    elif event["event"] == "done":
                        done = event
                break
            except WorkerError as exc:
                if not (exc.oom and attempt == 1):
                    raise
                batch_cap, max_length = 1, max(128, max_length // 2)
                yield {"phase": "train", "done": False,
                       "message": f"Out of memory; retrying with batch 1, max_length {max_length}"}

        assert done is not None
        label = _adapter_label(payload.get("goal") or plan.target_adapter_name or "model", run_id)
        archive = self._adapters / f"{label}-it{iteration}"
        await asyncio.to_thread(self._archive, Path(done["adapter_dir"]), archive, {
            "base_model": base_model_id,
            "goal": payload.get("goal", ""),
            "run_id": run_id,
            "iteration": iteration,
            "hyperparams": plan.hyperparams.model_dump(),
            "examples": len(plan.data_spec.examples),
            "train_loss": done.get("train_loss"),
            "peak_bytes": done.get("peak_bytes"),
        })
        yield {"phase": "archive", "done": False, "message": f"Saved adapter as {archive.name}"}
        yield {
            "done": True,
            "artifact": ArtifactMeta(
                run_id=run_id,
                iteration=iteration,
                adapter_path=str(archive),
                score=0.0,
                checksum=await asyncio.to_thread(_adapter_checksum, archive),
                is_mock=False,
            ),
            "cost_usd": 0.0,
            "duration_s": round(time.perf_counter() - started, 2),
        }

    @staticmethod
    def _archive(source: Path, target: Path, meta: dict) -> None:
        import shutil

        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        (target / "evolora.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class LocalModelRunner:
    """Answers prompts with the base model, optionally with a LoRA adapter applied."""

    is_mock = False

    def __init__(
        self,
        *,
        python: str,
        workspace: str | Path,
        base_model_id: str,
        max_new_tokens: int = 384,
    ) -> None:
        self._python = python
        self._workspace = Path(workspace).resolve()
        self._base_model_id = base_model_id
        self._max_new_tokens = max_new_tokens

    async def run_batch(self, prompts: list[dict], adapter_path: str | None = None) -> dict[str, str]:
        job_dir = self._workspace / "generate" / hashlib.sha1(
            f"{adapter_path}|{time.time_ns()}".encode()
        ).hexdigest()[:12]
        output = job_dir / "outputs.json"
        job = {
            "base_model": self._base_model_id,
            "adapter_dir": adapter_path,
            "prompts": prompts,
            "max_new_tokens": self._max_new_tokens,
            "output_file": str(output),
        }
        async for _ in run_worker(self._python, "generate", job, job_dir):
            pass
        return json.loads(output.read_text(encoding="utf-8"))
