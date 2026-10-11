"""On-device LoRA training and inference worker for the local profile.

Run as a standalone script by the training interpreter (``LOCAL_TRAIN_PYTHON``), not
imported by EvoLoRA: it needs torch/transformers/peft/trl, which the main environment
doesn't carry. Each call does one job and exits, so all GPU memory goes back to the OS.

    python worker.py train    job.json
    python worker.py generate job.json

Machine-readable output is one ``EVOLORA {json}`` line per event on stdout; anything
else is library noise. The recipe (TRL 0.24 SFT + PEFT, bf16, eager attention, gradient
checkpointing, non-fused AdamW) is what Phase 0 measured working on the Intel Arc 140V.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

MARKER = "EVOLORA "
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
OOM_SIGNS = ("out of memory", "out_of_resources", "level zero error 39", "ur_result_error_out_of")


def emit(event: str, **data) -> None:
    print(MARKER + json.dumps({"event": event, **data}), flush=True)


def pick_device():
    import torch

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", torch.bfloat16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def device_name(device: str) -> str:
    import torch

    if device == "xpu":
        return torch.xpu.get_device_name(0)
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"


def peak_memory(device: str) -> int:
    import torch

    if device == "xpu":
        return int(torch.xpu.max_memory_allocated())
    if device == "cuda":
        return int(torch.cuda.max_memory_allocated())
    return 0


def chat_text(tokenizer, prompt: str, completion: str | None = None) -> str:
    """Same chat formatting for training, evaluation and chat, so they match."""
    messages = [{"role": "user", "content": prompt}]
    if completion is None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    messages.append({"role": "assistant", "content": completion})
    return tokenizer.apply_chat_template(messages, tokenize=False)


def load_model(base_model: str, device: str, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=dtype, attn_implementation="eager"
    )
    return model.to(device), tokenizer


def train(job: dict) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    device, dtype = pick_device()
    started = time.perf_counter()
    model, tokenizer = load_model(job["base_model"], device, dtype)
    emit("loaded", device=device, device_name=device_name(device),
         load_s=round(time.perf_counter() - started, 2))

    hp = job["hyperparams"]
    rows = [{"text": chat_text(tokenizer, ex["prompt"], ex["completion"])} for ex in job["examples"]]
    output_dir = Path(job["output_dir"])

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                emit("progress", step=state.global_step, total_steps=state.max_steps,
                     loss=round(float(logs["loss"]), 4))

    config = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=hp["num_epochs"],
        per_device_train_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp["grad_accum"],
        learning_rate=hp["learning_rate"],
        optim="adamw_torch",
        bf16=dtype == torch.bfloat16,
        gradient_checkpointing=job.get("gradient_checkpointing", True),
        max_length=job["max_length"],
        dataset_text_field="text",
        dataset_num_proc=1,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,  # progress goes out as EVOLORA events instead
        dataloader_num_workers=0,
        seed=job.get("seed", 42),
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=hp["r"],
            lora_alpha=hp["lora_alpha"],
            target_modules=LORA_TARGETS,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainer.add_callback(Progress())
    # The default fused/foreach AdamW fails on the Arc 140V (Level Zero error 39).
    trainer.optimizer = torch.optim.AdamW(
        (p for p in trainer.model.parameters() if p.requires_grad),
        lr=hp["learning_rate"],
        foreach=False,
        fused=False,
    )
    train_started = time.perf_counter()
    result = trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    emit("done", adapter_dir=str(adapter_dir), train_loss=round(float(result.training_loss), 4),
         train_s=round(time.perf_counter() - train_started, 2), peak_bytes=peak_memory(device),
         steps=trainer.state.global_step)


def generate(job: dict) -> None:
    import torch

    device, dtype = pick_device()
    started = time.perf_counter()
    model, tokenizer = load_model(job["base_model"], device, dtype)
    if job.get("adapter_dir"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, job["adapter_dir"]).to(device)
    model.eval()
    # Greedy decoding: drop the model's sampling defaults so transformers doesn't warn
    # that temperature/top_p/top_k are ignored.
    gen_config = model.generation_config
    gen_config.do_sample = False
    gen_config.temperature = gen_config.top_p = gen_config.top_k = None
    emit("loaded", device=device, device_name=device_name(device),
         load_s=round(time.perf_counter() - started, 2))

    tokenizer.padding_side = "left"  # decoder-only models generate after the prompt
    prompts = job["prompts"]
    batch_size = max(1, int(job.get("batch_size", 4)))
    outputs: dict[str, str] = {}
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        texts = [chat_text(tokenizer, p["prompt"]) for p in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=job.get("max_new_tokens", 384),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, enc["input_ids"].shape[1]:]
        for p, tokens in zip(batch, new_tokens, strict=True):
            outputs[p["sample_id"]] = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        emit("progress", done=len(outputs), total=len(prompts))

    Path(job["output_file"]).write_text(json.dumps(outputs, ensure_ascii=False), encoding="utf-8")
    emit("done", count=len(outputs), peak_bytes=peak_memory(device))


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in {"train", "generate"}:
        print("usage: worker.py {train|generate} job.json", file=sys.stderr)
        return 64
    job = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    try:
        (train if sys.argv[1] == "train" else generate)(job)
    except Exception as exc:  # report, then fail: the backend decides whether to retry
        message = f"{type(exc).__name__}: {exc}"
        emit("error", error=message[:2000], oom=any(s in message.lower() for s in OOM_SIGNS))
        return 2
    finally:
        release_gpu_memory()
    return 0


def release_gpu_memory() -> None:
    """Hand cached GPU memory back before exiting so the LLM server can start right away."""
    import gc

    gc.collect()
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # best effort: process exit frees it anyway
        pass


if __name__ == "__main__":
    sys.exit(main())
