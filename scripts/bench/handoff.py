"""Measure llama-server startup, a local LoRA step, and warm restart."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def gpu_memory() -> dict:
    probe = subprocess.run(
        [sys.executable, "-c", "import json,torch; f,t=torch.xpu.mem_get_info();"
         "print(json.dumps({'free':f,'total':t}))"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(probe.stdout)


def start_server(binary: Path, model: Path, port: int, extra: list[str], log: Path):
    command = [str(binary), "-m", str(model), "--host", "127.0.0.1",
               "--port", str(port), "-ngl", "99", "-c", "4096", "--jinja", *extra]
    with log.open("wb") as output:
        started = time.perf_counter()
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    while time.perf_counter() - started < 180:
        if process.poll() is not None:
            raise RuntimeError(f"server exited {process.returncode}; see {log}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return process, time.perf_counter() - started
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(0.25)
    process.terminate()
    raise TimeoutError(f"server not ready within 180 s; see {log}")


def stop_server(process: subprocess.Popen) -> float:
    started = time.perf_counter()
    process.terminate()
    process.wait(timeout=20)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extra = ["--no-mmap"] if args.no_mmap else []
    result = {"binary": str(args.binary), "model": str(args.model), "extra": extra}
    result["before"] = gpu_memory()
    first, result["first_start_s"] = start_server(
        args.binary, args.model, args.port, extra, args.output.with_suffix(".first.log")
    )
    result["while_server"] = gpu_memory()
    result["first_stop_s"] = stop_server(first)
    result["after_first_stop"] = gpu_memory()
    env = os.environ.copy()
    env["HF_HOME"] = str(Path("models/hf").resolve())
    train_started = time.perf_counter()
    train = subprocess.run(
        [sys.executable, "scripts/bench/train_sft.py", "--steps", "3",
         "--gradient-checkpointing"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    result["train_s"] = time.perf_counter() - train_started
    result["train_returncode"] = train.returncode
    result["train_result"] = next(
        (line for line in train.stdout.splitlines() if line.startswith("BENCH_RESULT=")), None
    )
    result["train_error"] = train.stderr[-2000:] if train.returncode else None
    result["after_train"] = gpu_memory()
    second, result["warm_start_s"] = start_server(
        args.binary, args.model, args.port, extra, args.output.with_suffix(".warm.log")
    )
    result["while_warm_server"] = gpu_memory()
    result["warm_stop_s"] = stop_server(second)
    result["after_warm_stop"] = gpu_memory()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
