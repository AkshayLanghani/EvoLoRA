"""Stand-in for the training interpreter: answers worker jobs without torch.

Tests point the backend's WORKER at this file, so it runs as
``python -u fake_worker_python.py <command> <job.json>``. FAKE_WORKER_MODE picks the
behaviour: unset (success), ``fail``, or ``oom_once`` (first training attempt runs out
of memory).
"""

import json
import os
import sys
from pathlib import Path

MARKER = "EVOLORA "


def emit(event, **data):
    print(MARKER + json.dumps({"event": event, **data}), flush=True)


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "-u"]
    command, job_file = args[-2], Path(args[-1])
    job = json.loads(job_file.read_text(encoding="utf-8"))
    print("some library noise", flush=True)

    mode = os.environ.get("FAKE_WORKER_MODE", "")
    if mode == "fail":
        emit("error", error="RuntimeError: boom", oom=False)
        return 2

    if command == "train":
        flag = job_file.parent / "oom-once.flag"
        if mode == "oom_once" and not flag.exists():
            flag.write_text("x")
            emit("error", error="RuntimeError: UR error out of device memory", oom=True)
            return 2
        emit("loaded", device="xpu", device_name="Intel(R) Arc(TM) 140V GPU (16GB)", load_s=1.0)
        total = 3
        for step in range(1, total + 1):
            # Like tqdm, leave a progress bar on the same line as the event.
            print(f" {step}/{total} [00:01<00:00]", end="", flush=True)
            emit("progress", step=step, total_steps=total, loss=round(2.0 / step, 4))
        adapter = Path(job["output_dir"]) / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        (adapter / "adapter_config.json").write_text(json.dumps(job["hyperparams"]))
        emit("done", adapter_dir=str(adapter), train_loss=0.9, train_s=1.0, peak_bytes=123, steps=total)
        return 0

    outputs = {p["sample_id"]: f"answer to {p['prompt'][:20]}" for p in job["prompts"]}
    if job.get("adapter_dir"):
        outputs = {k: v + " (adapter)" for k, v in outputs.items()}
    Path(job["output_file"]).write_text(json.dumps(outputs), encoding="utf-8")
    emit("done", count=len(outputs), peak_bytes=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
