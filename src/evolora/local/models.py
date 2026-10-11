"""Catalog and downloader for the local LLMs that plan, judge and advise.

Base models for training are fetched separately by the training stack.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{filename}"
CHUNK = 1024 * 1024


@dataclass(frozen=True)
class LocalModel:
    key: str
    repo: str
    filename: str
    size_gb: float
    role: str
    note: str

    @property
    def url(self) -> str:
        return HF_RESOLVE.format(repo=self.repo, filename=self.filename)


# Sizes are the Q4_K_M files as published on Hugging Face.
CATALOG: dict[str, LocalModel] = {
    m.key: m
    for m in (
        LocalModel(
            key="qwen3-4b-instruct",
            repo="unsloth/Qwen3-4B-Instruct-2507-GGUF",
            filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            size_gb=2.33,
            role="planner (default)",
            note="Fast, strong tool calling for its size; leaves memory for training.",
        ),
        LocalModel(
            key="qwen3-8b",
            repo="Qwen/Qwen3-8B-GGUF",
            filename="Qwen3-8B-Q4_K_M.gguf",
            size_gb=4.68,
            role="planner (smarter, slower)",
            note="Better plans and evals; about half the speed of the 4B model.",
        ),
        LocalModel(
            key="phi-4-mini-instruct",
            repo="unsloth/Phi-4-mini-instruct-GGUF",
            filename="Phi-4-mini-instruct-Q4_K_M.gguf",
            size_gb=2.32,
            role="judge (independent)",
            note="A different model family, so the judge isn't grading its own plans.",
        ),
    )
}

DEFAULT_PLANNER = "qwen3-4b-instruct"


def model_path(models_dir: str | Path, model: LocalModel) -> Path:
    return Path(models_dir) / model.filename


def is_downloaded(models_dir: str | Path, model: LocalModel) -> bool:
    return model_path(models_dir, model).is_file()


def download(
    model: LocalModel,
    models_dir: str | Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Download ``model`` into ``models_dir``, resuming a partial ``.part`` file."""
    target = model_path(models_dir, model)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    done = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}

    with httpx.stream(
        "GET", model.url, headers=headers, follow_redirects=True, timeout=60.0
    ) as resp:
        if resp.status_code == 416:
            # Range rejection can also mean an oversized/corrupt partial. Only an
            # exact advertised size proves that this interrupted transfer finished.
            match = re.fullmatch(r"bytes \*/(\d+)", resp.headers.get("content-range", ""))
            if match and done > 0 and done == int(match[1]):
                partial.rename(target)
                return target
            raise RuntimeError("Model download resume rejected; partial size does not match remote file")
        resp.raise_for_status()
        if done and resp.status_code != 206:  # server ignored Range: start over
            done = 0
        length = resp.headers.get("content-length")
        total = done + int(length) if length is not None else 0
        if resp.status_code == 206:
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", resp.headers.get("content-range", "")
            )
            if not match or int(match[1]) != done or int(match[2]) < done:
                raise RuntimeError("Model download resume returned an unexpected byte range")
            total = int(match[3])
            if int(match[2]) >= total:
                raise RuntimeError("Model download resume returned an invalid total size")
        with partial.open("ab" if done else "wb") as fh:
            for chunk in resp.iter_bytes(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)
        if done == 0 or (total and done != total):
            raise RuntimeError(f"Model download incomplete: received {done} of {total} bytes")
    partial.rename(target)
    return target
