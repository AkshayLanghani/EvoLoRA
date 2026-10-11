"""Builds the planner, judge, retrain advisor and resource manager for the active profile.

Shared by the CLI, TUI and web app so the cloud and local profiles are wired the same way.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import httpx

from evolora.agent.planner import get_planner
from evolora.config import Config
from evolora.evaluation.digitalocean_judge import CandidateJudge, get_judge
from evolora.local.installer import python_has_torch
from evolora.local.models import CATALOG, model_path
from evolora.local.supervisor import LlamaServer, LocalResources, ModelSpec
from evolora.observability.run_logger import default_log_dir
from evolora.orchestration.retrain_advisor import RetrainAdvisor, get_retrain_advisor


def build_planner(cfg: Config):
    return get_planner(
        use_minimax=cfg.minimax_available,
        api_key=cfg.minimax_api_key,
        model=cfg.minimax_model,
        base_url=cfg.minimax_base_url,
        **cfg.planner_options(),
    )


def build_judge(cfg: Config) -> CandidateJudge:
    return get_judge(
        api_key=cfg.digital_ocean_model_access_key,
        base_url=cfg.digitalocean_inference_base_url,
        model=cfg.digital_ocean_judge_model,
        label=cfg.judge_provider,
    )


def build_retrain_advisor(cfg: Config) -> RetrainAdvisor:
    return get_retrain_advisor(
        api_key=cfg.minimax_api_key,
        model=cfg.minimax_model,
        base_url=cfg.minimax_base_url,
        label=cfg.llm_provider,
    )


def _resolve_model_path(cfg: Config, explicit: str, alias: str) -> str:
    """An explicit path wins; otherwise a catalog model downloaded by `evolora setup`."""
    if explicit:
        return explicit
    entry = CATALOG.get(alias)
    return str(model_path(cfg.local_models_dir, entry)) if entry else ""


def planner_spec(cfg: Config) -> ModelSpec:
    alias = cfg.local_llm_model
    return ModelSpec(path=_resolve_model_path(cfg, cfg.local_llm_model_path, alias), alias=alias)


def judge_spec(cfg: Config) -> ModelSpec | None:
    if not (cfg.local_judge_model or cfg.local_judge_model_path):
        return None
    alias = cfg.local_judge_model or cfg.local_llm_model
    return ModelSpec(path=_resolve_model_path(cfg, cfg.local_judge_model_path, alias), alias=alias)


@lru_cache(maxsize=1)
def _server_for(server_path: str, port: int, ctx: int, gpu_layers: int, extra: str) -> LlamaServer:
    # One server per process, shared by every run.
    return LlamaServer(
        server_path=server_path,
        port=port,
        ctx=ctx,
        gpu_layers=gpu_layers,
        extra_args=extra,
        log_path=Path(default_log_dir()) / "llama-server.log",
    )


def local_server(cfg: Config) -> LlamaServer | None:
    if not (cfg.is_local and cfg.local_llm_managed):
        return None
    port = urlparse(cfg.local_llm_base_url).port or 8080
    return _server_for(
        cfg.llama_server_path,
        port,
        cfg.local_llm_ctx,
        cfg.local_llm_gpu_layers,
        cfg.local_llm_extra_args,
    )


def local_llm_issue(cfg: Config) -> str | None:
    """Why the local LLM can't serve right now, or None when it can (or isn't used).

    Checked once before a run: calling a server that can't exist would cost minutes of
    client retries before each stage fell back anyway.
    """
    if not cfg.is_local:
        return None
    if not cfg.local_llm_managed:
        try:
            ok = httpx.get(f"{cfg.local_llm_base_url.rstrip('/')}/models", timeout=2).status_code == 200
        except httpx.HTTPError:
            ok = False
        return None if ok else f"no server reachable at {cfg.local_llm_base_url}"
    if not cfg.llama_server_path or not Path(cfg.llama_server_path).is_file():
        return "llama-server not found (set LLAMA_SERVER_PATH)"
    for role, spec in (("planner", planner_spec(cfg)), ("judge", judge_spec(cfg))):
        if spec is not None and not (spec.path and Path(spec.path).is_file()):
            return f"{role} model {spec.alias!r} not downloaded (run `evolora setup`)"
    return None


def local_training_issue(cfg: Config) -> str | None:
    """Why on-device training can't start, or None. `evolora doctor` checks the libraries."""
    if not (cfg.is_local and cfg.training_backend == "local"):
        return None
    if not Path(cfg.local_train_python).is_file():
        return f"training interpreter not found at {cfg.local_train_python!r} (set LOCAL_TRAIN_PYTHON)"
    if not python_has_torch(cfg.local_train_python):
        return "training environment not installed (press Install everything)"
    return None


def prepare_run(cfg: Config) -> tuple[Config, list[str]]:
    """The config a run should use, plus notices to show at its start.

    Parts of the local stack that aren't ready switch to their labeled fallbacks up front:
    heuristics for the LLM stages, the mock trainer for training.
    """
    updates: dict = {}
    notices: list[str] = []
    if issue := local_llm_issue(cfg):
        updates |= {"minimax_api_key": "", "digital_ocean_model_access_key": ""}
        notices.append(f"Local LLM not ready: {issue}. Using the heuristic planner, judge and advisor.")
    if issue := local_training_issue(cfg):
        updates |= {"training_backend": "mock", "model_runner": "mock"}
        notices.append(f"Local training not ready: {issue}. Using the mock trainer.")
    return (cfg.model_copy(update=updates), notices) if updates else (cfg, [])


def build_resources(cfg: Config, *, backend_uses_gpu: bool = False) -> LocalResources | None:
    """A fresh manager per run (its warnings are per run); the server itself is shared."""
    if not cfg.is_local or not cfg.minimax_available:  # heuristic fallback: no LLM to manage
        return None
    return LocalResources(
        base_url=cfg.local_llm_base_url,
        planner=planner_spec(cfg),
        judge=judge_spec(cfg),
        server=local_server(cfg),
        release_for_gpu=backend_uses_gpu,
    )
