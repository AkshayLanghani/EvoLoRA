"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

PROFILES = ("cloud", "local")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
# Backends the local profile may use: on-device training, or mock for a quick dry run.
LOCAL_TRAINING_BACKENDS = {"local", "mock"}

# Base models small enough to LoRA-train on a 16 GB laptop GPU (measured in Phase 0:
# Qwen2.5-1.5B bf16 LoRA peaks at ~3 GB on the Intel Arc 140V).
LOCAL_BASE_MODELS = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5 1.5B (recommended)",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 0.5B (fastest)",
}
CLOUD_BASE_MODELS = {
    "unsloth/Phi-3-mini-4k-instruct": "Phi-3-mini (fast)",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B",
}


def _default_train_python() -> str:
    """The training interpreter: a dedicated venv if present, else this interpreter."""
    import sys

    for candidate in (".venv-train", ".venv-bench"):
        path = Path(candidate) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if path.is_file():
            return str(path.resolve())
    return sys.executable


class ConfigError(ValueError):
    """The environment asks for a combination the selected profile does not allow."""


def is_loopback_url(url: str) -> bool:
    return (urlparse(url).hostname or "") in LOOPBACK_HOSTS


class Config(BaseModel):
    # cloud: hosted services (MiniMax, DigitalOcean, MongoDB, LiveKit, remote GPU VM).
    # local: everything on this machine; LLM calls go to a local llama.cpp server.
    profile: str = Field(default="cloud")

    # Local LLM server (llama.cpp llama-server, OpenAI-compatible)
    local_llm_base_url: str = Field(default="http://127.0.0.1:8080/v1")
    local_llm_model: str = Field(default="qwen3-4b-instruct")
    local_judge_model: str = Field(default="")  # blank = same model as the planner
    local_llm_timeout_s: float = Field(default=900.0)
    local_llm_max_tokens: int = Field(default=4096)
    local_llm_example_batch_size: int = Field(default=8)
    local_llm_managed: bool = Field(default=True)  # False = you run the server yourself
    llama_server_path: str = Field(default="")
    local_llm_model_path: str = Field(default="")
    local_judge_model_path: str = Field(default="")
    local_llm_ctx: int = Field(default=8192)  # 16k overflowed the Arc 140V's shared memory after training
    local_llm_gpu_layers: int = Field(default=99)
    local_llm_extra_args: str = Field(default="")
    local_models_dir: str = Field(default="./models")
    local_tools_dir: str = Field(default="./tools")  # llama.cpp builds installed by setup

    # On-device training (local profile)
    local_train_python: str = Field(default="")
    local_workspace_dir: str = Field(default="./workspace")
    local_adapters_dir: str = Field(default="./adapters")
    local_max_seq_len: int = Field(default=512)
    local_max_batch: int = Field(default=4, ge=1, le=32)
    local_max_new_tokens: int = Field(default=384)

    # Local run history (SQLite); used by the local profile or whenever set.
    db_path: str = Field(default="")

    # MiniMax
    minimax_api_key: str = Field(default="")
    minimax_model: str = Field(default="MiniMax-M2.7-highspeed")
    minimax_base_url: str = Field(default="https://api.minimax.io/v1")
    minimax_group_id: str = Field(default="")

    # Training
    training_backend: str = Field(default="mock")  # mock | unsloth | remote
    model_runner: str = Field(default="mock")       # mock | local | remote
    base_model_id: str = Field(default="unsloth/Phi-3-mini-4k-instruct")

    # Remote GPU config push
    ssh_host: str = Field(default="")
    ssh_user: str = Field(default="")
    ssh_port: int = Field(default=22)
    ssh_key_path: str = Field(default="")
    remote_config_path: str = Field(default="/workspace/config.json")
    remote_results_path: str = Field(default="/workspace/generations/results.json")

    # Persistence
    mongodb_uri: str = Field(default="")
    mongodb_db_name: str = Field(default="evolora")
    mongodb_runs_collection: str = Field(default="runs")
    mongodb_server_selection_timeout_ms: int = Field(default=3000)
    artifact_store: str = Field(default="local")
    artifact_dir: str = Field(default="./artifacts")

    # Loop control
    max_iterations: int = Field(default=3)
    target_score: float = Field(default=0.85)
    improvement_threshold: float = Field(default=0.01)
    patience: int = Field(default=2)
    # When true, the run is fully autonomous: no user approvals (eval set + keep-training
    # gates are skipped), it iterates until the target/judge is satisfied or max iterations.
    auto_approve: bool = Field(default=False)

    # DigitalOcean
    digitalocean_inference_base_url: str = Field(
        default="https://inference.do-ai.run/v1/"
    )
    digital_ocean_model_access_key: str = Field(default="")
    digital_ocean_judge_model: str = Field(default="llama3.3-70b-instruct")
    digitalocean_token: str = Field(default="")

    # Voice (LiveKit Inference STT/TTS) — optional, fully decoupled from training.
    livekit_url: str = Field(default="")
    livekit_api_key: str = Field(default="")
    livekit_api_secret: str = Field(default="")
    voice_enabled: bool = Field(default=True)
    stt_model: str = Field(default="deepgram/nova-3")
    tts_model: str = Field(default="cartesia/sonic-3")
    tts_voice: str = Field(default="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
    narrate_interval: float = Field(default=30.0)
    narrate_polish: bool = Field(default=True)  # rephrase template lines via MiniMax when available
    ptt_key: str = Field(default="f9")  # global push-to-talk key (pynput name)

    @property
    def mock_mode(self) -> bool:
        return self.training_backend == "mock"

    @property
    def is_local(self) -> bool:
        return self.profile == "local"

    @property
    def base_models(self) -> dict[str, str]:
        """Selectable base models: small trainable ones locally, the VM models in the cloud."""
        return LOCAL_BASE_MODELS if self.is_local else CLOUD_BASE_MODELS

    @property
    def default_base_model(self) -> str:
        models = self.base_models
        return self.base_model_id if self.base_model_id in models else next(iter(models))

    @property
    def llm_provider(self) -> str:
        """Display label for whoever answers planner/advisor calls."""
        if self.is_local:
            return "local LLM"
        return "MiniMax" if self.minimax_available else "heuristic"

    @property
    def judge_provider(self) -> str:
        if self.is_local:
            return "local LLM"
        return "DigitalOcean" if self.digital_ocean_judge_available else "heuristic"

    def planner_options(self) -> dict:
        """Extra planner settings; local models are slower and handle smaller batches."""
        if not self.is_local:
            return {}
        return {
            "timeout": self.local_llm_timeout_s,
            "tool_max_tokens": self.local_llm_max_tokens,
            "eval_max_tokens": self.local_llm_max_tokens,
            "example_batch_size": self.local_llm_example_batch_size,
            "label": "local LLM",
            # llama.cpp only accepts string tool_choice values (Phase 0 finding).
            "named_tool_choice": False,
        }

    @property
    def minimax_available(self) -> bool:
        return bool(self.minimax_api_key)

    @property
    def mongo_available(self) -> bool:
        return bool(self.mongodb_uri)

    @property
    def digital_ocean_judge_available(self) -> bool:
        return bool(self.digital_ocean_model_access_key)

    @property
    def voice_available(self) -> bool:
        """Voice can run only when enabled and all LiveKit creds are present.

        Package import + audio-device availability are checked at runtime by the
        VoiceService; this just gates on config so the TUI can decide whether to try.
        """
        return bool(
            self.voice_enabled
            and self.livekit_url
            and self.livekit_api_key
            and self.livekit_api_secret
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _apply_local_profile(cfg: Config) -> Config:
    """Route every service to this machine and refuse anything that would leave it.

    The existing LLM clients (planner, judge, retrain advisor, narrator) all speak the
    OpenAI API with a configurable base URL, so pointing them at llama-server is enough.
    """
    if not is_loopback_url(cfg.local_llm_base_url):
        raise ConfigError(
            f"LOCAL_LLM_BASE_URL must point at this machine (got {cfg.local_llm_base_url!r})"
        )
    if cfg.training_backend not in LOCAL_TRAINING_BACKENDS:
        raise ConfigError(
            f"TRAINING_BACKEND={cfg.training_backend!r} is not available in the local profile "
            f"(allowed: {', '.join(sorted(LOCAL_TRAINING_BACKENDS))})"
        )
    if cfg.model_runner not in LOCAL_TRAINING_BACKENDS:
        raise ConfigError(f"MODEL_RUNNER={cfg.model_runner!r} is not available in the local profile")

    return cfg.model_copy(
        update={
            # llama-server ignores the key; the clients only need it to be non-empty.
            "minimax_api_key": "local",
            "minimax_base_url": cfg.local_llm_base_url,
            "minimax_model": cfg.local_llm_model,
            "digital_ocean_model_access_key": "local",
            "digitalocean_inference_base_url": cfg.local_llm_base_url,
            "digital_ocean_judge_model": cfg.local_judge_model or cfg.local_llm_model,
            "mongodb_uri": "",
            "voice_enabled": False,  # LiveKit is a hosted service
            "db_path": cfg.db_path or "./evolora.db",
            "local_train_python": cfg.local_train_python or _default_train_python(),
            "llama_server_path": cfg.llama_server_path or find_llama_server(cfg.local_tools_dir),
        }
    )


def find_llama_server(tools_dir: str, name: str | None = None) -> str:
    """A llama-server installed by `evolora agent` setup, preferring GPU builds."""
    root = Path(tools_dir)
    name = name or ("llama-server.exe" if os.name == "nt" else "llama-server")
    for flavor in ("metal", "sycl", "cuda", "vulkan", "cpu"):
        # "**" because the macOS archive nests the binary one folder down.
        for exe in sorted(root.glob(f"llama-*/{flavor}/**/{name}"), reverse=True):
            return str(exe.resolve())
    return ""


@lru_cache(maxsize=1)
def get_config() -> Config:
    profile = os.getenv("EVOLORA_PROFILE", "cloud").strip().lower()
    if profile not in PROFILES:
        raise ConfigError(f"EVOLORA_PROFILE must be one of {PROFILES} (got {profile!r})")
    cfg = _load_config(profile)
    return _apply_local_profile(cfg) if cfg.is_local else cfg


def _load_config(profile: str) -> Config:
    # The local profile trains on this machine unless told otherwise; cloud defaults to mock.
    default_backend = "local" if profile == "local" else "mock"
    return Config(
        profile=profile,
        local_train_python=os.getenv("LOCAL_TRAIN_PYTHON", ""),
        local_workspace_dir=os.getenv("LOCAL_WORKSPACE_DIR", "./workspace"),
        local_adapters_dir=os.getenv("LOCAL_ADAPTERS_DIR", "./adapters"),
        local_max_seq_len=int(os.getenv("LOCAL_MAX_SEQ_LEN", "512")),
        local_max_batch=int(os.getenv("LOCAL_MAX_BATCH", "4")),
        local_max_new_tokens=int(os.getenv("LOCAL_MAX_NEW_TOKENS", "384")),
        local_llm_base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
        local_llm_model=os.getenv("LOCAL_LLM_MODEL", "qwen3-4b-instruct"),
        local_judge_model=os.getenv("LOCAL_JUDGE_MODEL", ""),
        local_llm_timeout_s=float(os.getenv("LOCAL_LLM_TIMEOUT_S", "900")),
        local_llm_max_tokens=int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096")),
        local_llm_example_batch_size=int(os.getenv("LOCAL_LLM_EXAMPLE_BATCH_SIZE", "8")),
        local_llm_managed=_env_bool("LOCAL_LLM_MANAGED", True),
        llama_server_path=os.getenv("LLAMA_SERVER_PATH", ""),
        local_llm_model_path=os.getenv("LOCAL_LLM_MODEL_PATH", ""),
        local_judge_model_path=os.getenv("LOCAL_JUDGE_MODEL_PATH", ""),
        local_llm_ctx=int(os.getenv("LOCAL_LLM_CTX", "8192")),
        local_llm_gpu_layers=int(os.getenv("LOCAL_LLM_GPU_LAYERS", "99")),
        local_llm_extra_args=os.getenv("LOCAL_LLM_EXTRA_ARGS", ""),
        local_models_dir=os.getenv("LOCAL_MODELS_DIR", "./models"),
        local_tools_dir=os.getenv("LOCAL_TOOLS_DIR", "./tools"),
        db_path=os.getenv("EVOLORA_DB_PATH", ""),
        minimax_api_key=os.getenv("MINIMAX_API_KEY", ""),
        minimax_model=os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"),
        minimax_base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
        minimax_group_id=os.getenv("MINIMAX_GROUP_ID", ""),
        training_backend=os.getenv("TRAINING_BACKEND", default_backend),
        model_runner=os.getenv("MODEL_RUNNER", default_backend),
        ssh_host=os.getenv("SSH_HOST", ""),
        ssh_user=os.getenv("SSH_USER", ""),
        ssh_port=int(os.getenv("SSH_PORT", "22")),
        ssh_key_path=os.getenv("SSH_KEY_PATH", ""),
        remote_config_path=os.getenv("REMOTE_CONFIG_PATH", "/workspace/config.json"),
        remote_results_path=os.getenv(
            "REMOTE_RESULTS_PATH", "/workspace/generations/results.json"
        ),
        mongodb_uri=os.getenv("MONGODB_URI", ""),
        mongodb_db_name=os.getenv("MONGODB_DB_NAME", "evolora"),
        mongodb_runs_collection=os.getenv("MONGODB_RUNS_COLLECTION", "runs"),
        mongodb_server_selection_timeout_ms=int(
            os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "3000")
        ),
        artifact_store=os.getenv("ARTIFACT_STORE", "local"),
        artifact_dir=os.getenv("ARTIFACT_DIR", "./artifacts"),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
        target_score=float(os.getenv("TARGET_SCORE", "0.85")),
        improvement_threshold=float(os.getenv("IMPROVEMENT_THRESHOLD", "0.01")),
        patience=int(os.getenv("PATIENCE", "2")),
        digitalocean_inference_base_url=os.getenv(
            "DIGITAL_OCEAN_INFERENCE_BASE_URL", "https://inference.do-ai.run/v1/"
        ),
        digital_ocean_model_access_key=os.getenv("DIGITAL_OCEAN_MODEL_ACCESS_KEY", ""),
        digital_ocean_judge_model=os.getenv(
            "DIGITAL_OCEAN_JUDGE_MODEL", "llama3.3-70b-instruct"
        ),
        base_model_id=os.getenv("BASE_MODEL_ID", "unsloth/Phi-3-mini-4k-instruct"),
        digitalocean_token=os.getenv("DIGITALOCEAN_TOKEN", ""),
        auto_approve=os.getenv("AUTO_APPROVE", "false").strip().lower() in {"1", "true", "yes", "on"},
        livekit_url=os.getenv("LIVEKIT_URL", ""),
        livekit_api_key=os.getenv("LIVEKIT_API_KEY", ""),
        livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
        voice_enabled=os.getenv("VOICE_ENABLED", "true").strip().lower()
        not in {"0", "false", "off", "no"},
        stt_model=os.getenv("STT_MODEL", "deepgram/nova-3"),
        tts_model=os.getenv("TTS_MODEL", "cartesia/sonic-3"),
        tts_voice=os.getenv("TTS_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        narrate_interval=float(os.getenv("NARRATE_INTERVAL", "30")),
        narrate_polish=os.getenv("NARRATE_POLISH", "true").strip().lower()
        not in {"0", "false", "off", "no"},
        ptt_key=os.getenv("PTT_KEY", "f9"),
    )
