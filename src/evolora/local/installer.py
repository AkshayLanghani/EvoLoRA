"""One-click setup for the local profile on Windows and Apple Silicon Macs.

Detects the GPU, then installs what a fully local run needs, each from its official
source: a llama.cpp server build (GitHub releases), the planner GGUF (Hugging Face), a
training environment (PyTorch for the detected GPU via uv), and the default base model.
Setup runs as one background job at a time; the web UI polls its progress.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from evolora.config import Config, find_llama_server, get_config
from evolora.local.models import CATALOG, DEFAULT_PLANNER, download, is_downloaded

LLAMA_CPP_TAG = "b11552"  # the build measured in Phase 0
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"
# Preferred llama.cpp build per GPU vendor; CUDA builds also need the runtime DLLs.
LLAMA_CPP_BUILDS: dict[str, list[str]] = {
    "intel": ["llama-{tag}-bin-win-sycl-x64.zip"],
    "nvidia": ["llama-{tag}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"],
    "other": ["llama-{tag}-bin-win-vulkan-x64.zip"],
    # macOS: the arm64 build uses Metal on Apple Silicon.
    "apple": ["llama-{tag}-bin-macos-arm64.tar.gz"],
}
LLAMA_CPP_FLAVOR = {"intel": "sycl", "nvidia": "cuda", "other": "vulkan", "apple": "metal"}
TORCH_VERSION = "2.14.1"
# "default" = the plain PyPI wheel, which on Apple Silicon already includes the MPS backend.
TORCH_BACKEND = {"intel": "xpu", "nvidia": "auto", "other": "cpu", "apple": "default"}
LLAMA_SERVER_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"
REQUIREMENTS = Path(__file__).with_name("requirements-train.txt")
TRAIN_OK_MARKER = ".evolora-ok"


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


@dataclass
class Hardware:
    gpus: list[str]
    ram_gb: float
    vendor: str  # intel | nvidia | apple | other

    def as_dict(self) -> dict:
        return {"gpus": self.gpus, "ram_gb": self.ram_gb, "vendor": self.vendor}


def classify_gpus(names: list[str]) -> str:
    joined = " ".join(names).lower()
    if "nvidia" in joined:
        return "nvidia"
    if "intel" in joined and ("arc" in joined or "iris" in joined):
        return "intel"
    return "other"


def _sysctl(name: str) -> str:
    out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10)
    return out.stdout.strip()


def detect_mac_hardware() -> Hardware:
    """Apple Silicon has one integrated GPU that shares system memory; name it by chip."""
    try:
        ram_gb = round(float(_sysctl("hw.memsize") or 0) / 2**30, 1)
        chip = _sysctl("machdep.cpu.brand_string")
    except (OSError, subprocess.SubprocessError, ValueError):
        ram_gb, chip = 0.0, ""
    if platform.machine() == "arm64":
        return Hardware(gpus=[f"{chip or 'Apple Silicon'} GPU"], ram_gb=ram_gb, vendor="apple")
    return Hardware(gpus=[], ram_gb=ram_gb, vendor="other")


def detect_hardware() -> Hardware:
    """GPU names and RAM: CIM on Windows, sysctl on macOS. Falls back to 'other' elsewhere."""
    if sys.platform == "darwin":
        return detect_mac_hardware()
    script = (
        "$g=@(Get-CimInstance Win32_VideoController | ForEach-Object Name);"
        "$m=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory;"
        "@{gpus=$g;ram=$m} | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        data = json.loads(out.stdout)
        gpus = [g for g in (data.get("gpus") or []) if g]
        ram_gb = round(float(data.get("ram") or 0) / 2**30, 1)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        gpus, ram_gb = [], 0.0
    return Hardware(gpus=gpus, ram_gb=ram_gb, vendor=classify_gpus(gpus))


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def hf_env() -> dict[str, str]:
    """Environment for processes that download from Hugging Face.

    The 10 s default read timeout was too short for multi-GB weight files on home
    connections (seen as repeated "Read timed out" retries).
    """
    env = dict(os.environ)
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return env


def train_python_path(base: str | Path = ".venv-train") -> Path:
    return Path(base) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def hf_cache_has(repo: str) -> bool:
    """Whether ``repo``'s weights are fully downloaded (not just its folder or config)."""
    hub = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_dir = hub / f"models--{repo.replace('/', '--')}"
    if any((model_dir / "blobs").glob("*.incomplete")):
        return False
    return any(
        w.is_file() and w.stat().st_size > 1_000_000  # follows symlinks into blobs/
        for w in (model_dir / "snapshots").glob("*/*.safetensors")
    )


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

Report = Callable[[str, float | None], None]  # (message, fraction done or None)


@dataclass
class Component:
    key: str
    title: str
    size: str
    installed: Callable[[Config], bool]
    install: Callable[[Config, Hardware, Report], None]


def _download_file(url: str, target: Path, report: Report, label: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with partial.open("wb") as fh:
            for chunk in resp.iter_bytes(1024 * 1024):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    report(f"Downloading {label}: {done / 2**20:.0f} / {total / 2**20:.0f} MB", done / total)
    partial.replace(target)


def _extract(archive: Path, dest: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")  # refuses paths that escape dest


def _install_llama_server(cfg: Config, hw: Hardware, report: Report) -> None:
    if sys.platform == "darwin" and hw.vendor != "apple":
        raise RuntimeError("EvoLoRA's local mode needs an Apple Silicon Mac (M1 or newer)")
    flavor = LLAMA_CPP_FLAVOR[hw.vendor]
    dest = Path(cfg.local_tools_dir) / f"llama-{LLAMA_CPP_TAG}" / flavor
    downloads = Path(cfg.local_tools_dir) / "downloads"
    for pattern in LLAMA_CPP_BUILDS[hw.vendor]:
        asset = pattern.format(tag=LLAMA_CPP_TAG)
        archive = downloads / asset
        if not archive.is_file():
            _download_file(LLAMA_CPP_URL.format(tag=LLAMA_CPP_TAG, asset=asset), archive, report, asset)
        report(f"Extracting {asset}", None)
        _extract(archive, dest)
    # Windows zips are flat; the macOS tarball nests the binaries in a folder.
    server = next(iter(sorted(dest.rglob(LLAMA_SERVER_NAME))), None)
    if server is None:
        raise RuntimeError(f"{LLAMA_SERVER_NAME} missing after extracting into {dest}")
    if os.name != "nt":
        server.chmod(0o755)
    report(f"llama.cpp {LLAMA_CPP_TAG} ({flavor}) installed", 1.0)


def _planner_model(cfg: Config):
    return CATALOG.get(cfg.local_llm_model) or CATALOG[DEFAULT_PLANNER]


def _install_planner(cfg: Config, hw: Hardware, report: Report) -> None:
    model = _planner_model(cfg)
    download(
        model,
        cfg.local_models_dir,
        lambda done, total: report(
            f"Downloading {model.filename}: {done / 2**20:.0f} / {total / 2**20:.0f} MB",
            done / total if total else None,
        ),
    )
    report(f"{model.key} downloaded", 1.0)


def _run(cmd: list[str], report: Report, label: str) -> None:
    report(label, None)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
        env=hf_env(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.strip():
            report(line.strip()[:200], None)
    if proc.wait() != 0:
        raise RuntimeError(f"{label} failed (exit {proc.returncode})")


def _install_training_env(cfg: Config, hw: Hardware, report: Report) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv not found on PATH (re-run the EvoLoRA installer)")
    python = train_python_path()
    if not python.is_file():
        _run([uv, "venv", ".venv-train", "--python", "3.11"], report, "Creating .venv-train")
    backend = TORCH_BACKEND[hw.vendor]
    torch_cmd = [uv, "pip", "install", "--python", str(python), f"torch=={TORCH_VERSION}"]
    if backend != "default":
        torch_cmd += ["--torch-backend", backend]
    _run(torch_cmd, report, f"Installing PyTorch {TORCH_VERSION} ({backend})")
    _run([uv, "pip", "install", "--python", str(python), "-r", str(REQUIREMENTS)],
         report, "Installing transformers, peft, trl")
    _run([str(python), "-c", "import torch, transformers, peft, trl; print('training env ok')"],
         report, "Checking the training environment")
    (python.parent.parent / TRAIN_OK_MARKER).write_text("ok", encoding="utf-8")
    report("Training environment ready", 1.0)


def _install_base_model(cfg: Config, hw: Hardware, report: Report) -> None:
    repo = cfg.default_base_model
    code = f"from huggingface_hub import snapshot_download; snapshot_download({repo!r})"
    python = train_python_path() if train_python_path().is_file() else Path(cfg.local_train_python)
    _run([str(python), "-c", code], report, f"Downloading {repo}")
    report(f"{repo} downloaded", 1.0)


def python_has_torch(python: str | Path) -> bool:
    """Whether ``python``'s environment has torch, without starting it (that takes seconds)."""
    path = Path(python)
    if not path.is_file():
        return False
    if path.resolve() == Path(sys.executable).resolve():
        return importlib.util.find_spec("torch") is not None
    venv = path.parent.parent
    return (venv / "Lib" / "site-packages" / "torch").is_dir() or any(
        venv.glob("lib/python3*/site-packages/torch")
    )


def _training_env_ready(cfg: Config) -> bool:
    """Set up by us (marker), or another environment the config points at that has torch."""
    own = train_python_path()
    if own.is_file() and (own.parent.parent / TRAIN_OK_MARKER).is_file():
        return True
    return python_has_torch(cfg.local_train_python)


COMPONENTS: list[Component] = [
    Component(
        "llama_server", "Local LLM server (llama.cpp)", "30–630 MB",
        lambda cfg: bool(find_llama_server(cfg.local_tools_dir)), _install_llama_server,
    ),
    Component(
        "planner_model", "Planner model (Qwen3-4B, GGUF)", "2.3 GB",
        lambda cfg: is_downloaded(cfg.local_models_dir, _planner_model(cfg)), _install_planner,
    ),
    Component(
        "training_env", "Training environment (PyTorch + PEFT/TRL)", "3–5 GB",
        _training_env_ready, _install_training_env,
    ),
    Component(
        "base_model", "Base model to fine-tune (Qwen2.5-1.5B)", "3.1 GB",
        lambda cfg: hf_cache_has(cfg.default_base_model), _install_base_model,
    ),
]
COMPONENT_KEYS = [c.key for c in COMPONENTS]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@dataclass
class SetupJob:
    components: list[str]
    status: str = "running"  # running | done | failed
    current: str = ""
    progress: float | None = None
    log: list[str] = field(default_factory=list)
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "components": self.components,
            "status": self.status,
            "current": self.current,
            "progress": self.progress,
            "log": self.log[-80:],
            "error": self.error,
        }


class SetupManager:
    """Runs one setup job at a time in a worker thread."""

    def __init__(self, cfg_factory: Callable[[], Config]) -> None:
        self._cfg = cfg_factory
        self._hardware: Hardware | None = None
        self.job: SetupJob | None = None

    @property
    def hardware(self) -> Hardware:
        if self._hardware is None:
            self._hardware = detect_hardware()
        return self._hardware

    def status(self) -> dict:
        cfg = self._cfg()
        return {
            "hardware": self.hardware.as_dict(),
            "components": [
                {"key": c.key, "title": c.title, "size": c.size, "installed": c.installed(cfg)}
                for c in COMPONENTS
            ],
            "job": self.job.as_dict() if self.job else None,
        }

    def start(self, keys: list[str] | None = None) -> SetupJob:
        if self.job and self.job.status == "running":
            raise RuntimeError("Setup is already running")
        cfg = self._cfg()
        wanted = keys or [c.key for c in COMPONENTS if not c.installed(cfg)]
        unknown = [k for k in wanted if k not in COMPONENT_KEYS]
        if unknown:
            raise ValueError(f"Unknown setup component(s): {', '.join(unknown)}")
        self.job = SetupJob(components=wanted)
        asyncio.get_running_loop().run_in_executor(None, self._run, self.job, cfg)
        return self.job

    def _run(self, job: SetupJob, cfg: Config) -> None:
        def report(message: str, fraction: float | None) -> None:
            job.progress = fraction
            if not job.log or job.log[-1] != message:
                if message.startswith("Downloading") and job.log and job.log[-1].startswith("Downloading"):
                    job.log[-1] = message  # keep one updating line per download
                else:
                    job.log.append(message)

        try:
            for component in COMPONENTS:
                if component.key not in job.components:
                    continue
                job.current, job.progress = component.key, None
                job.log.append(f"== {component.title}")
                component.install(cfg, self.hardware, report)
            job.status, job.current = "done", ""
            job.log.append("Setup complete")
            get_config.cache_clear()  # pick up the new llama-server and training env paths
        except Exception as exc:  # surface in the UI; the user can retry
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            job.log.append(f"Failed: {job.error}")
