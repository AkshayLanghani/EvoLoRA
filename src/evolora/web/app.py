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

import asyncio
import importlib.metadata
import json
import os
import tempfile
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware


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

from evolora.agent.tools import (  # noqa: E402
    ALPHA_MULTIPLIERS,
    BATCH_SIZES,
    LEARNING_RATES,
    LORA_RANKS,
    NUM_TRAIN_EPOCHS,
    coerce_hyperparams,
)
from evolora.config import Config, get_config  # noqa: E402
from evolora.demo.task import ADAPTIVE_EVAL_SET, LOCKED_EVAL_SET  # noqa: E402
from evolora.evaluation.evaluator import evaluate_sample  # noqa: E402
from evolora.evaluation.locked import LockedEvalSet  # noqa: E402
from evolora.local.installer import SetupManager  # noqa: E402
from evolora.models.core import EvalSample, RunConfig, RunRecord  # noqa: E402
from evolora.models.events import Event, EventKind  # noqa: E402
from evolora.orchestration.orchestrator import Orchestrator  # noqa: E402
from evolora.persistence.artifacts import LocalArtifactStore  # noqa: E402
from evolora.persistence.store import RunStore, get_run_store  # noqa: E402
from evolora.services import (  # noqa: E402
    build_judge,
    build_planner,
    build_resources,
    build_retrain_advisor,
    local_llm_issue,
    local_server,
    local_training_issue,
    prepare_run,
)
from evolora.training.backends import _adapter_label, get_backend  # noqa: E402
from evolora.training.runner import get_runner  # noqa: E402
from evolora.web.remote_access import PairingStore, RemoteAccessMiddleware  # noqa: E402

PUBLIC_DIR = Path(__file__).resolve().parents[3] / "public"

MAX_GOAL_CHARS = 500


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


# Public deployments spend the owner's API credits, so web runs are capped. The local
# profile serves only this machine and uses the loop settings as configured.
WEB_MAX_ITERATIONS = _env_int("WEB_MAX_ITERATIONS", 3)
WEB_MAX_TRAINING_SAMPLES = _env_int("WEB_MAX_TRAINING_SAMPLES", 50)
WEB_RUNS_PER_IP_PER_HOUR = _env_int("WEB_RUNS_PER_IP_PER_HOUR", 10)
WEB_MAX_CONCURRENT_RUNS = _env_int("WEB_MAX_CONCURRENT_RUNS", 3)
LOCAL_MAX_TRAINING_SAMPLES = 1000
MIN_USEFUL_EXAMPLES = 5  # below this a real LoRA run barely moves (seen in practice)


def _max_iterations(cfg: Config) -> int:
    return cfg.max_iterations if cfg.is_local else min(cfg.max_iterations, WEB_MAX_ITERATIONS)


def _max_training_samples(cfg: Config) -> int:
    return LOCAL_MAX_TRAINING_SAMPLES if cfg.is_local else WEB_MAX_TRAINING_SAMPLES


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
_active_chat = False
_store: RunStore | None = None
# Runs streaming right now, so approvals and cancels can reach them. Only meaningful in
# a long-lived process (the local agent); serverless instances may not share it.
_live_runs: dict[str, Orchestrator] = {}


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


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    cfg = get_config()
    if cfg.is_local:
        store = _run_store()
        if hasattr(store, "mark_interrupted"):  # runs cut off by the last shutdown
            await store.mark_interrupted()
    yield
    if cfg.is_local and (server := local_server(cfg)) is not None:
        await server.stop()  # never leave llama-server holding GPU memory


app = FastAPI(
    lifespan=_lifespan,
    title="EvoLoRA",
    description="Run the bounded self-improvement loop over HTTP.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url=None,
)

_pairings = PairingStore(
    os.getenv("EVOLORA_PAIRINGS_PATH", str(Path.home() / ".evolora" / "pairings.json"))
)
_setup = SetupManager(get_config)

if get_config().is_local:
    # A hosted EvoLoRA site may drive this agent only with a token the user approved.
    app.add_middleware(RemoteAccessMiddleware, store=_pairings)
    # The local agent listens on loopback only; rejecting other Host headers also stops
    # DNS-rebinding pages from driving it through the browser.
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"]
    )


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    return origin is None or origin == f"http://{request.headers.get('host', '')}"


# ---------------------------------------------------------------------------
# Agent discovery, pairing and one-click setup (local agent)
# ---------------------------------------------------------------------------


@app.get("/api/agent/hello")
async def agent_hello(request: Request) -> dict[str, Any]:
    """Lets a hosted EvoLoRA site detect this agent and whether it is paired."""
    _require_local()
    origin = request.headers.get("origin", "")
    return {"app": "evolora-agent", "version": _version(), "paired": _pairings.is_paired(origin)}


class PairRequest(BaseModel):
    origin: str = Field(max_length=300)


@app.post("/api/pair")
async def pair(req: PairRequest, request: Request) -> dict[str, str]:
    """Approve a website. Only the agent's own pairing page may call this."""
    _require_local()
    if not _same_origin(request):
        raise HTTPException(403, "Pairing must be approved on this computer")
    try:
        token = _pairings.pair(req.origin)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"origin": req.origin, "token": token}


@app.get("/api/pairings")
async def list_pairings(request: Request) -> dict[str, list[str]]:
    _require_local()
    if not _same_origin(request):
        raise HTTPException(403, "Only available on this computer")
    return {"origins": _pairings.origins()}


@app.delete("/api/pairings")
async def revoke_pairing(origin: str, request: Request) -> dict[str, Any]:
    _require_local()
    if not _same_origin(request):
        raise HTTPException(403, "Only available on this computer")
    return {"origin": origin, "revoked": _pairings.revoke(origin)}


@app.get("/api/setup")
async def setup_status() -> dict[str, Any]:
    """Hardware, what is installed, and the current setup job."""
    _require_local()
    return await asyncio.to_thread(_setup.status)


class SetupRequest(BaseModel):
    components: list[str] | None = None  # None = everything not yet installed


@app.post("/api/setup/install")
async def setup_install(req: SetupRequest) -> dict[str, Any]:
    """One click: download and install what a fully local run needs."""
    _require_local()
    try:
        job = _setup.start(req.components)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return job.as_dict()


def _version() -> str:
    try:
        return importlib.metadata.version("evolora")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@app.get("/api/resources")
async def resource_status() -> dict[str, Any]:
    """Live host resources for local runs; do not expose hosted server telemetry."""
    if not get_config().is_local:
        return {"available": False}
    from evolora.local.resources import system_resources

    return {"available": True, **await asyncio.to_thread(system_resources)}


@app.get("/api/status")
async def status() -> dict[str, Any]:
    """Which live services are configured. Never returns secret values."""
    cfg = get_config()
    issue = local_llm_issue(cfg)
    if cfg.is_local and issue:
        planner, judge = "heuristic", "heuristic"
    elif cfg.is_local:
        planner, judge = "local", "local"
    else:
        planner = "minimax" if cfg.minimax_available else "heuristic"
        judge = "digitalocean" if cfg.digital_ocean_judge_available else "heuristic"
    body: dict[str, Any] = {
        "mode": "local" if cfg.is_local else "hosted",
        "planner": planner,
        "planner_model": cfg.minimax_model if planner != "heuristic" else None,
        "judge": judge,
        "judge_model": cfg.digital_ocean_judge_model if judge != "heuristic" else None,
        "persistence": _persistence_label(cfg),
        "training_backend": cfg.training_backend,
        "model_runner": cfg.model_runner,
        "approvals": cfg.is_local,
        "base_models": [{"id": k, "label": v} for k, v in cfg.base_models.items()],
        "default_base_model": cfg.default_base_model,
        "loop": {
            "max_iterations": _max_iterations(cfg),
            "target_score": cfg.target_score,
            "improvement_threshold": cfg.improvement_threshold,
            "patience": cfg.patience,
        },
        "limits": {
            "max_goal_chars": MAX_GOAL_CHARS,
            "max_training_samples": _max_training_samples(cfg),
            "runs_per_ip_per_hour": None if cfg.is_local else WEB_RUNS_PER_IP_PER_HOUR,
        },
    }
    if cfg.is_local:
        resources = build_resources(cfg)
        body["local_llm"] = resources.status() if resources else None
        body["local_llm_issue"] = issue
        training_issue = local_training_issue(cfg)
        body["local_training_issue"] = training_issue
        if training_issue:
            body["training_backend"] = "mock"
    return body


def _persistence_label(cfg: Config) -> str:
    if cfg.db_path:
        return "sqlite"
    return "mongodb" if cfg.mongo_available else "memory"


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    goal: str = Field(default="", max_length=MAX_GOAL_CHARS)
    base_model_id: str | None = None  # None = the profile's default base model
    training_samples: int = Field(default=10, ge=1)
    # Only the local agent can pause for approvals: it is one long-lived process, so the
    # approval request reaches the same orchestrator. Hosted runs are always autonomous.
    autonomous: bool = True


def _build_orchestrator(req: RunRequest) -> Orchestrator:
    cfg, notices = prepare_run(get_config())
    backend = get_backend(cfg.training_backend)
    base_model_id = req.base_model_id or cfg.default_base_model
    if not backend.is_mock and req.training_samples < MIN_USEFUL_EXAMPLES:
        notices.append(
            f"Only {req.training_samples} training example(s) requested; LoRA needs at least "
            f"{MIN_USEFUL_EXAMPLES} (ideally 20+) to learn anything measurable."
        )
    run_config = RunConfig(
        max_iterations=_max_iterations(cfg),
        target_score=cfg.target_score,
        improvement_threshold=cfg.improvement_threshold,
        patience=cfg.patience,
        training_backend=cfg.training_backend,
        model_runner=cfg.model_runner,
        base_model_id=base_model_id,
        training_sample_count=req.training_samples,
        goal=req.goal.strip(),
        require_retrain_approval=cfg.is_local and not req.autonomous,
    )
    return Orchestrator(
        config=run_config,
        eval_set=LOCKED_EVAL_SET,
        planner=build_planner(cfg),
        training_backend=backend,
        model_runner=get_runner(cfg.model_runner, base_model_id=base_model_id),
        run_store=_run_store(),
        artifact_store=LocalArtifactStore(cfg.artifact_dir),
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=build_judge(cfg),
        retrain_advisor=build_retrain_advisor(cfg),
        resources=build_resources(cfg, backend_uses_gpu=not backend.is_mock),
        notices=notices,
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
    run_id = ""
    try:
        async with aclosing(await orch.run()) as events:
            async for event in events:
                if not run_id:
                    run_id = event.run_id
                    _live_runs[run_id] = orch
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
                "ts": datetime.now(UTC).isoformat(),
            }
        )
    finally:
        # Reached on normal completion and when the client disconnects (generator closed).
        orch.cancel()
        _live_runs.pop(run_id, None)


class _RunResponse(StreamingResponse):
    """Hold the run slot until streaming finishes, including client disconnects."""

    async def __call__(self, scope, receive, send) -> None:
        global _active_runs
        try:
            await super().__call__(scope, receive, send)
        finally:
            _active_runs -= 1


@app.post("/api/runs")
async def start_run(req: RunRequest, request: Request) -> StreamingResponse:
    """Start a run and stream its events as NDJSON until it finishes."""
    global _active_runs
    cfg = get_config()
    if req.base_model_id is not None and req.base_model_id not in cfg.base_models:
        raise HTTPException(
            422, f"Unknown base model. Choose one of: {', '.join(cfg.base_models)}"
        )
    cap = _max_training_samples(cfg)
    if req.training_samples > cap:
        raise HTTPException(422, f"training_samples is capped at {cap} here")
    # Locally, runs share one GPU and one LLM server: run them one at a time.
    max_concurrent = 1 if cfg.is_local else WEB_MAX_CONCURRENT_RUNS
    if _active_runs >= max_concurrent or (cfg.is_local and _active_chat):
        raise HTTPException(503, "Another run is in progress. Try again when it finishes.")
    if not cfg.is_local and not _limiter.allow(_client_ip(request)):
        raise HTTPException(
            429, f"Limit of {WEB_RUNS_PER_IP_PER_HOUR} runs per hour reached. Try again later."
        )
    try:
        orch = _build_orchestrator(req)
    except (ImportError, ValueError, RuntimeError) as exc:
        raise HTTPException(503, f"Training backend unavailable: {exc}") from exc

    # No await between admission and reservation: another request cannot reserve the
    # same local GPU before the response body starts running.
    _active_runs += 1
    return _RunResponse(
        _stream_run(orch),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/runs")
async def list_runs(limit: int = 10) -> dict[str, Any]:
    """Recent runs. Persistent with SQLite or MongoDB; otherwise only this instance's memory."""
    records = await _run_store().list_runs(limit=max(1, min(limit, 50)))
    return {
        "persistence": _persistence_label(get_config()),
        "runs": [_record_summary(r) for r in records],
    }


class ApprovalRequest(BaseModel):
    approved: bool


def _live_run(run_id: str) -> Orchestrator:
    orch = _live_runs.get(run_id)
    if orch is None:
        raise HTTPException(404, f"No run {run_id!r} is in progress on this server")
    return orch


@app.post("/api/runs/{run_id}/approval")
async def answer_approval(run_id: str, req: ApprovalRequest) -> dict[str, Any]:
    """Answer a pending approval gate (approve evals, or keep training)."""
    _live_run(run_id).submit_retrain_approval(req.approved)
    return {"run_id": run_id, "approved": req.approved}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, Any]:
    _live_run(run_id).cancel()
    return {"run_id": run_id, "cancelled": True}


# ---------------------------------------------------------------------------
# Adapters and chat (local agent)
# ---------------------------------------------------------------------------


def _require_local() -> Config:
    cfg = get_config()
    if not cfg.is_local:
        raise HTTPException(404, "Only available in the local agent")
    return cfg


def _adapter_dirs(cfg: Config) -> dict[str, Path]:
    root = Path(cfg.local_adapters_dir)
    if not root.is_dir():
        return {}
    return {p.name: p for p in root.iterdir() if (p / "evolora.json").is_file()}


@app.get("/api/adapters")
async def list_adapters() -> dict[str, Any]:
    """Adapters trained on this machine, newest first."""
    cfg = _require_local()
    adapters = []
    for name, path in _adapter_dirs(cfg).items():
        meta = json.loads((path / "evolora.json").read_text(encoding="utf-8"))
        adapters.append({"name": name, "modified": path.stat().st_mtime, **meta})
    adapters.sort(key=lambda a: a["modified"], reverse=True)
    return {"adapters": adapters}


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    adapter: str | None = None  # None = the base model alone
    base_model_id: str | None = None
    # An adapter only behaves as trained when it sees the instruction it was trained with.
    apply_goal: bool = True


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    """One-shot answer from a trained adapter (or a base model), run on this machine."""
    global _active_chat
    cfg = _require_local()
    if _active_runs or _active_chat:
        raise HTTPException(503, "The GPU is in use. Chat when the current operation finishes.")
    adapter_path: str | None = None
    base_model = req.base_model_id or cfg.default_base_model
    prompt = req.prompt
    goal_applied = False
    if req.adapter:
        path = _adapter_dirs(cfg).get(req.adapter)
        if path is None:
            raise HTTPException(404, f"No adapter named {req.adapter!r}")
        adapter_path = str(path)
        meta = json.loads((path / "evolora.json").read_text(encoding="utf-8"))
        base_model = meta["base_model"]
        goal = str(meta.get("goal") or "").strip()
        if req.apply_goal and goal and not prompt.lstrip().startswith(goal):
            # Same shape as the locked eval prompts the adapter was scored on.
            prompt = f"{goal}\n\nInput:\n{prompt}"
            goal_applied = True
    elif base_model not in cfg.base_models:
        raise HTTPException(422, f"Unknown base model {base_model!r}")
    if issue := local_training_issue(cfg):
        raise HTTPException(503, f"Local inference not ready: {issue}")

    _active_chat = True
    try:
        resources = build_resources(cfg, backend_uses_gpu=True)
        if resources is not None:
            await resources.acquire("gpu")  # free the LLM's memory first
        runner = get_runner("local", base_model_id=base_model)
        started = time.perf_counter()
        answer = await runner.run_batch([{"sample_id": "chat", "prompt": prompt}], adapter_path)
        return {
            "answer": answer.get("chat", ""),
            "adapter": req.adapter,
            "goal_applied": goal_applied,
            "base_model": base_model,
            "seconds": round(time.perf_counter() - started, 1),
        }
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    finally:
        _active_chat = False


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


class _RevalidatedStaticFiles(StaticFiles):
    """Static files the browser must revalidate before reuse.

    Without a Cache-Control header browsers guess a freshness lifetime, so after an update
    the page could load new HTML with an old cached stylesheet. ``no-cache`` still allows
    cheap 304 responses through the ETag.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response


# Locally, serve the site from the same server. On Vercel, public/ is served by the CDN.
if PUBLIC_DIR.is_dir():
    app.mount("/", _RevalidatedStaticFiles(directory=PUBLIC_DIR, html=True), name="site")
