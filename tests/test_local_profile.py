"""Tests for the local profile: service routing, loopback-only guard, wiring."""

import pytest

from evolora.agent.planner import MiniMaxPlanner, get_planner
from evolora.config import ConfigError, get_config
from evolora.local.supervisor import LocalResources
from evolora.persistence.sqlite_store import SqliteRunStore
from evolora.persistence.store import get_run_store
from evolora.services import (
    build_judge,
    build_resources,
    build_retrain_advisor,
    local_llm_issue,
    local_training_issue,
    prepare_run,
)

_KEYS = (
    "EVOLORA_PROFILE",
    "LOCAL_LLM_BASE_URL",
    "LOCAL_LLM_MODEL",
    "LOCAL_JUDGE_MODEL",
    "TRAINING_BACKEND",
    "MODEL_RUNNER",
    "MINIMAX_API_KEY",
    "MONGODB_URI",
    "EVOLORA_DB_PATH",
    "LLAMA_SERVER_PATH",
    "LOCAL_LLM_MODEL_PATH",
    "LOCAL_LLM_MANAGED",
    "LOCAL_TRAIN_PYTHON",
)


@pytest.fixture
def env(monkeypatch, tmp_path):
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EVOLORA_DB_PATH", str(tmp_path / "runs.db"))
    # Isolate from anything installed in the repo (tools/, models/).
    monkeypatch.setenv("LOCAL_TOOLS_DIR", str(tmp_path / "tools"))
    monkeypatch.setenv("LOCAL_MODELS_DIR", str(tmp_path / "models"))
    get_config.cache_clear()
    yield monkeypatch
    get_config.cache_clear()


def _local(env, **extra):
    env.setenv("EVOLORA_PROFILE", "local")
    for key, value in extra.items():
        env.setenv(key, value)
    get_config.cache_clear()
    return get_config()


def test_cloud_is_default_and_unchanged(env):
    cfg = get_config()
    assert cfg.profile == "cloud"
    assert not cfg.is_local
    assert cfg.minimax_base_url == "https://api.minimax.io/v1"
    assert cfg.planner_options() == {}


def test_local_profile_routes_every_llm_to_loopback(env):
    env.setenv("MINIMAX_API_KEY", "real-cloud-key")
    env.setenv("MONGODB_URI", "mongodb+srv://example")
    cfg = _local(env, LOCAL_LLM_MODEL="qwen3-8b")

    assert cfg.minimax_base_url == "http://127.0.0.1:8080/v1"
    assert cfg.minimax_api_key == "local"  # the cloud key is never sent anywhere
    assert cfg.minimax_model == "qwen3-8b"
    assert cfg.digitalocean_inference_base_url == "http://127.0.0.1:8080/v1"
    assert cfg.digital_ocean_judge_model == "qwen3-8b"
    assert cfg.mongodb_uri == ""
    assert cfg.voice_enabled is False
    assert cfg.llm_provider == cfg.judge_provider == "local LLM"


def test_local_judge_model_can_differ(env):
    cfg = _local(env, LOCAL_JUDGE_MODEL="phi-4-mini-instruct")
    assert cfg.digital_ocean_judge_model == "phi-4-mini-instruct"
    assert cfg.minimax_model == "qwen3-4b-instruct"


def test_local_profile_rejects_remote_llm_url(env):
    with pytest.raises(ConfigError, match="this machine"):
        _local(env, LOCAL_LLM_BASE_URL="https://api.example.com/v1")


def test_local_profile_rejects_cloud_training(env):
    with pytest.raises(ConfigError, match="TRAINING_BACKEND"):
        _local(env, TRAINING_BACKEND="remote")


def test_unknown_profile_is_rejected(env):
    env.setenv("EVOLORA_PROFILE", "hybrid")
    get_config.cache_clear()
    with pytest.raises(ConfigError):
        get_config()


def test_local_planner_gets_local_limits(env):
    cfg = _local(env)
    planner = get_planner(
        use_minimax=cfg.minimax_available,
        api_key=cfg.minimax_api_key,
        model=cfg.minimax_model,
        base_url=cfg.minimax_base_url,
        **cfg.planner_options(),
    )
    assert isinstance(planner, MiniMaxPlanner)
    assert planner.label == "local LLM"
    assert planner._timeout == cfg.local_llm_timeout_s
    assert planner._example_batch_size == cfg.local_llm_example_batch_size
    assert planner._base_url == "http://127.0.0.1:8080/v1"


def test_local_judge_and_advisor_are_labeled(env):
    cfg = _local(env)
    assert build_judge(cfg).label == "local LLM"
    assert build_retrain_advisor(cfg).label == "local LLM"


def test_resources_only_in_local_profile(env):
    assert build_resources(get_config()) is None
    assert isinstance(build_resources(_local(env)), LocalResources)


def test_unready_local_llm_falls_back_up_front(env):
    cfg = _local(env)  # no LLAMA_SERVER_PATH, no model downloaded
    assert "llama-server not found" in local_llm_issue(cfg)

    run_cfg, notices = prepare_run(cfg)
    assert not run_cfg.minimax_available
    assert not run_cfg.digital_ocean_judge_available
    assert "heuristic" in notices[0]
    assert build_resources(run_cfg) is None


def test_ready_local_llm_is_used(env, tmp_path):
    server = tmp_path / "llama-server.exe"
    model = tmp_path / "planner.gguf"
    server.write_bytes(b"")
    model.write_bytes(b"")
    cfg = _local(env, LLAMA_SERVER_PATH=str(server), LOCAL_LLM_MODEL_PATH=str(model))

    assert local_llm_issue(cfg) is None
    run_cfg, notices = prepare_run(cfg)
    assert run_cfg is cfg and notices == []


def test_training_env_without_torch_uses_mock_trainer(env, tmp_path):
    venv = tmp_path / "venv-no-torch"
    python = venv / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    cfg = _local(env, LOCAL_TRAIN_PYTHON=str(python))

    assert "not installed" in local_training_issue(cfg)
    run_cfg, notices = prepare_run(cfg)
    assert run_cfg.training_backend == "mock" and run_cfg.model_runner == "mock"
    assert any("mock trainer" in n for n in notices)

    (venv / "Lib" / "site-packages" / "torch").mkdir(parents=True)
    assert local_training_issue(cfg) is None


def test_cloud_profile_never_reports_local_issues(env):
    cfg = get_config()
    assert local_llm_issue(cfg) is None
    assert prepare_run(cfg) == (cfg, [])


def test_db_path_selects_sqlite_store(env):
    assert isinstance(get_run_store(get_config()), SqliteRunStore)
