"""FastAPI app that runs the real EvoLoRA orchestrator for browser clients.

Deployed as a Vercel Python function (see ``api/index.py``) and runnable locally with
``uv run uvicorn evolora.web.app:app --reload``, which also serves the ``public/`` site.

Runs are streamed to the browser as NDJSON, one orchestrator event per line. A browser
can't answer an approval gate in the middle of a streamed response, so web runs are
always autonomous (the AUTO_APPROVE behaviour). Service selection follows the same
environment variables as the CLI/TUI: MiniMax, the DigitalOcean judge, MongoDB and the
remote GPU backend switch on when their keys are set, otherwise the labeled fallbacks run.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, TypeAdapter, ValidationError


def _prepare_environment() -> None:
    """Adapt filesystem-dependent settings before get_config() is first called."""
    if os.getenv("VERCEL"):
        # Serverless functions can only write under the temp dir.
        tmp = Path(tempfile.gettempdir()) / "evolora"
        os.environ.setdefault("EVOLORA_LOG_DIR", str(tmp / "logs"))
        os.environ.setdefault("ARTIFACT_DIR", str(tmp / "artifacts"))

    # Hosting platforms store secrets as env vars, not files: materialize the key the
    # remote backend expects at SSH_KEY_PATH.
    key = os.getenv("SSH_PRIVATE_KEY", "")
    if key and not os.getenv("SSH_KEY_PATH"):
        path = Path(tempfile.gettempdir()) / "evolora_ssh_key"
        path.write_text(key.replace("\\n", "\n").strip() + "\n", encoding="utf-8")
        path.chmod(0o600)
        os.environ["SSH_KEY_PATH"] = str(path)


_prepare_environment()

from evolora.agent.planner import get_planner  # noqa: E402
from evolora.agent.tools import (  # noqa: E402
    ALPHA_MULTIPLIERS,
    BATCH_SIZES,
    LEARNING_RATES,
    LORA_RANKS,
    NUM_TRAIN_EPOCHS,
    coerce_hyperparams,
)
from evolora.config import get_config  # noqa: E402
from evolora.demo.task import ADAPTIVE_EVAL_SET, LOCKED_EVAL_SET  # noqa: E402
from evolora.evaluation.digitalocean_judge import get_judge  # noqa: E402
from evolora.evaluation.evaluator import evaluate_sample  # noqa: E402
from evolora.evaluation.locked import LockedEvalSet  # noqa: E402
from evolora.models.core import EvalSample, RunConfig, RunRecord  # noqa: E402
from evolora.models.events import Event, EventKind  # noqa: E402
from evolora.orchestration.orchestrator import Orchestrator  # noqa: E402
from evolora.orchestration.retrain_advisor import get_retrain_advisor  # noqa: E402
from evolora.persistence.artifacts import LocalArtifactStore  # noqa: E402
from evolora.persistence.store import RunStore, get_run_store  # noqa: E402
from evolora.training.backends import _adapter_label, get_backend  # noqa: E402
from evolora.training.runner import get_runner  # noqa: E402

PUBLIC_DIR = Path(__file__).resolve().parents[3] / "public"

# Same choices as the TUI's base-model dropdown.
BASE_MODELS: dict[str, str] = {
    "unsloth/Phi-3-mini-4k-instruct": "Phi-3-mini (fast)",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B",
}

MAX_GOAL_CHARS = 500


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


# Public deployments spend the owner's API credits, so web runs are capped.
WEB_MAX_ITERATIONS = _env_int("WEB_MAX_ITERATIONS", 3)
WEB_MAX_TRAINING_SAMPLES = _env_int("WEB_MAX_TRAINING_SAMPLES", 50)
WEB_RUNS_PER_IP_PER_HOUR = _env_int("WEB_RUNS_PER_IP_PER_HOUR", 10)
WEB_MAX_CONCURRENT_RUNS = _env_int("WEB_MAX_CONCURRENT_RUNS", 3)


class _RateLimiter:
    """Sliding-window limiter. Per process, so on serverless it is best-effort."""

    def __init__(self, limit: int, window_s: float) -> None:
        self._limit = limit
        self._window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self._window:
            hits.popleft()
        if len(hits) >= self._limit:
            return False
        hits.append(now)
        return True


_limiter = _RateLimiter(WEB_RUNS_PER_IP_PER_HOUR, 3600)
_active_runs = 0
_store: RunStore | None = None


def _run_store() -> RunStore:
    global _store
    if _store is None:
        _store = get_run_store()
    return _store


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


app = FastAPI(
    title="EvoLoRA",
    description="Run the bounded self-improvement loop over HTTP.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url=None,
)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@app.get("/api/status")
async def status() -> dict[str, Any]:
    """Which live services are configured. Never returns secret values."""
    cfg = get_config()
    return {
        "planner": "minimax" if cfg.minimax_available else "heuristic",
        "planner_model": cfg.minimax_model if cfg.minimax_available else None,
        "judge": "digitalocean" if cfg.digital_ocean_judge_available else "heuristic",
        "judge_model": cfg.digital_ocean_judge_model if cfg.digital_ocean_judge_available else None,
        "persistence": "mongodb" if cfg.mongo_available else "memory",
        "training_backend": cfg.training_backend,
        "model_runner": cfg.model_runner,
        "base_models": [{"id": k, "label": v} for k, v in BASE_MODELS.items()],
        "default_base_model": (
            cfg.base_model_id if cfg.base_model_id in BASE_MODELS else next(iter(BASE_MODELS))
        ),
        "loop": {
            "max_iterations": min(cfg.max_iterations, WEB_MAX_ITERATIONS),
            "target_score": cfg.target_score,
            "improvement_threshold": cfg.improvement_threshold,
            "patience": cfg.patience,
        },
        "limits": {
            "max_goal_chars": MAX_GOAL_CHARS,
            "max_training_samples": WEB_MAX_TRAINING_SAMPLES,
            "runs_per_ip_per_hour": WEB_RUNS_PER_IP_PER_HOUR,
        },
    }


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    goal: str = Field(default="", max_length=MAX_GOAL_CHARS)
    base_model_id: str = Field(default="unsloth/Phi-3-mini-4k-instruct")
    training_samples: int = Field(default=10, ge=1)


def _build_orchestrator(req: RunRequest) -> Orchestrator:
    cfg = get_config()
    run_config = RunConfig(
        max_iterations=min(cfg.max_iterations, WEB_MAX_ITERATIONS),
        target_score=cfg.target_score,
        improvement_threshold=cfg.improvement_threshold,
        patience=cfg.patience,
        training_backend=cfg.training_backend,
        model_runner=cfg.model_runner,
        base_model_id=req.base_model_id,
        training_sample_count=req.training_samples,
        goal=req.goal.strip(),
        # Nobody can answer an approval gate mid-stream, so web runs are autonomous.
        require_retrain_approval=False,
    )
    return Orchestrator(
        config=run_config,
        eval_set=LOCKED_EVAL_SET,
        planner=get_planner(
            use_minimax=cfg.minimax_available,
            api_key=cfg.minimax_api_key,
            model=cfg.minimax_model,
            base_url=cfg.minimax_base_url,
        ),
        training_backend=get_backend(cfg.training_backend),
        model_runner=get_runner(cfg.model_runner),
        run_store=_run_store(),
        artifact_store=LocalArtifactStore(cfg.artifact_dir),
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=get_judge(
            api_key=cfg.digital_ocean_model_access_key,
            base_url=cfg.digitalocean_inference_base_url,
            model=cfg.digital_ocean_judge_model,
        ),
        retrain_advisor=get_retrain_advisor(
            api_key=cfg.minimax_api_key,
            model=cfg.minimax_model,
            base_url=cfg.minimax_base_url,
        ),
    )


def _ndjson(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, default=str) + "\n").encode()


def _event_payload(event: Event) -> dict[str, Any]:
    return event.model_dump(mode="python")


def _record_summary(record: RunRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "goal": record.config.goal,
        "base_model_id": record.config.base_model_id,
        "status": record.status.value,
        "stop_reason": record.stop_reason.value if record.stop_reason else None,
        "eval_set_hash": record.eval_set_hash,
        "baseline_score": record.baseline_score,
        "best_score": record.best_score,
        "best_iteration": record.best_iteration,
        "iteration_scores": [it.score for it in record.iterations],
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "adapter_name": _adapter_label(record.config.goal or "model", record.run_id),
    }


async def _stream_run(orch: Orchestrator) -> AsyncIterator[bytes]:
    global _active_runs
    _active_runs += 1
    run_id = ""
    try:
        async for event in await orch.run():
            run_id = event.run_id
            yield _ndjson(_event_payload(event))
        record = await _run_store().get(run_id) if run_id else None
        if record is not None:
            yield _ndjson({"kind": "run_summary", "run_id": run_id, "data": _record_summary(record)})
    except Exception as exc:  # surface failures to the browser instead of a cut-off stream
        yield _ndjson(
            {
                "kind": EventKind.RUN_FAILED.value,
                "run_id": run_id,
                "message": f"Server error: {type(exc).__name__}: {exc}",
                "data": {"reason": "server_error"},
            }
        )
    finally:
        # Reached on normal completion and when the client disconnects (generator closed).
        orch.cancel()
        _active_runs -= 1


@app.post("/api/runs")
async def start_run(req: RunRequest, request: Request) -> StreamingResponse:
    """Start a run and stream its events as NDJSON until it finishes."""
    if req.base_model_id not in BASE_MODELS:
        raise HTTPException(422, f"Unknown base model. Choose one of: {', '.join(BASE_MODELS)}")
    if req.training_samples > WEB_MAX_TRAINING_SAMPLES:
        raise HTTPException(422, f"training_samples is capped at {WEB_MAX_TRAINING_SAMPLES} here")
    if _active_runs >= WEB_MAX_CONCURRENT_RUNS:
        raise HTTPException(503, "The server is busy with other runs. Try again in a minute.")
    if not _limiter.allow(_client_ip(request)):
        raise HTTPException(
            429, f"Limit of {WEB_RUNS_PER_IP_PER_HOUR} runs per hour reached. Try again later."
        )
    try:
        orch = _build_orchestrator(req)
    except (ImportError, ValueError, RuntimeError) as exc:
        raise HTTPException(503, f"Training backend unavailable: {exc}") from exc

    return StreamingResponse(
        _stream_run(orch),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/runs")
async def list_runs(limit: int = 10) -> dict[str, Any]:
    """Recent runs. Persistent with MONGODB_URI; otherwise only this instance's memory."""
    records = await _run_store().list_runs(limit=max(1, min(limit, 50)))
    return {
        "persistence": "mongodb" if get_config().mongo_available else "memory",
        "runs": [_record_summary(r) for r in records],
    }


# ---------------------------------------------------------------------------
# Guardrail tools: the real Python checks, exposed one at a time
# ---------------------------------------------------------------------------


class SnapRequest(BaseModel):
    args: dict[str, Any]


@app.post("/api/tools/snap")
async def snap(req: SnapRequest) -> dict[str, Any]:
    """Snap proposed start_training_model arguments onto the allowed choice sets."""
    return {
        "snapped": coerce_hyperparams(req.args),
        "choices": {
            "learning_rate": LEARNING_RATES,
            "lora_rank_r": LORA_RANKS,
            "lora_alpha_multiplier": ALPHA_MULTIPLIERS,
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "per_device_train_batch_size": BATCH_SIZES,
        },
    }


@app.get("/api/tools/eval-set")
async def eval_set() -> dict[str, Any]:
    """The demo task's locked eval set and its SHA-256 hash.

    ``samples_json`` is Python-serialized text. Clients edit and return that text
    rather than re-serializing, since JavaScript would turn 550.0 into 550 and change
    the hash of an untouched set.
    """
    samples = [s.model_dump() for s in LOCKED_EVAL_SET.samples]
    return {
        "hash": LOCKED_EVAL_SET.hash,
        "samples": samples,
        "samples_json": json.dumps(samples, indent=2),
    }


class VerifyRequest(BaseModel):
    samples_json: str = Field(max_length=20_000)


_SAMPLES_ADAPTER = TypeAdapter(list[EvalSample])


@app.post("/api/tools/verify-evals")
async def verify_evals(req: VerifyRequest) -> dict[str, Any]:
    """Hash a submitted eval set and compare it with the locked one."""
    try:
        samples = _SAMPLES_ADAPTER.validate_json(req.samples_json)
    except ValidationError as exc:
        raise HTTPException(422, f"Not a valid eval set: {exc.errors()[0]['msg']}") from exc
    computed = LockedEvalSet(samples).hash
    return {
        "locked_hash": LOCKED_EVAL_SET.hash,
        "computed_hash": computed,
        "match": computed == LOCKED_EVAL_SET.hash,
    }


class ScoreRequest(BaseModel):
    sample_id: str
    response: str = Field(max_length=5000)


@app.post("/api/tools/score")
async def score(req: ScoreRequest) -> dict[str, Any]:
    """Score a model response against a locked sample with the objective evaluator."""
    sample = next((s for s in LOCKED_EVAL_SET.samples if s.sample_id == req.sample_id), None)
    if sample is None:
        raise HTTPException(404, f"No locked sample {req.sample_id!r}")
    try:
        result = evaluate_sample(sample, req.response)
    except AttributeError:
        # evaluate_sample expects a JSON object; arrays/scalars don't have .keys().
        return {"score": 0.0, "passed": False, "details": {"error": "response is not a JSON object"}}
    return result.model_dump(include={"score", "passed", "details"})


class NameRequest(BaseModel):
    goal: str = Field(max_length=MAX_GOAL_CHARS)
    run_id: str = Field(default="", max_length=64)


@app.post("/api/tools/adapter-name")
async def adapter_name(req: NameRequest) -> dict[str, str]:
    """The archive name a run with this goal would save its adapter under."""
    run_id = req.run_id or uuid.uuid4().hex
    return {"name": _adapter_label(req.goal, run_id), "run_id": run_id}


# Locally, serve the site from the same server. On Vercel, public/ is served by the CDN.
if PUBLIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="site")
