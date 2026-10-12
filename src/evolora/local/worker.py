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
import math
import os
import sys
import time
from pathlib import Path

MARKER = "EVOLORA "
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
OOM_SIGNS = ("out of memory", "out_of_resources", "level zero error 39", "ur_result_error_out_of")


def validate_training_metrics(logs: dict) -> None:
    """Trainer's default NaN filter can make a broken run look like zero loss."""
    for key in ("loss", "grad_norm", "entropy"):
        if key in logs and not math.isfinite(float(logs[key])):
            raise RuntimeError(f"Numerically unstable training: non-finite {key}; adapter not saved")
    if float(logs.get("grad_norm", 0)) > 1e6:
        raise RuntimeError("Numerically unstable training: exploding gradients; adapter not saved")


def validate_trainable_tensors(model, torch, *, gradients: bool = False) -> None:
    """Reject invalid updates before AdamW, and invalid weights before saving."""
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        tensor = parameter.grad if gradients else parameter
        if tensor is not None and not torch.isfinite(tensor).all().item():
            kind = "gradient" if gradients else "weight"
            raise RuntimeError(f"Numerically unstable training: non-finite {kind} {name}")


def clip_xpu_gradients(parameters, max_norm: float, norm_type: float = 2.0):
    """Use CPU reductions for adapter norms to avoid unstable XPU reduction kernels."""
    import torch

    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    norms = [torch.linalg.vector_norm(g.detach().to(device="cpu", dtype=torch.float32), norm_type)
             for g in grads]
    total = torch.linalg.vector_norm(torch.stack(norms), norm_type)
    validate_training_metrics({"grad_norm": total.item()})
    coefficient = min(1.0, max_norm / (total.item() + 1e-6))
    with torch.no_grad():
        for grad in grads:
            grad.mul_(coefficient)
    return total


def composed_softmax(input, dim=None, _stacklevel=3, dtype=None):
    """Softmax from elementary ops, with the signature of ``torch.nn.functional.softmax``.

    On the Arc 140V (PyTorch 2.14.1+xpu) the native softmax *backward* is wrong once the
    softmax dimension exceeds 256: the forward values are right but gradients come out
    hundreds to tens of thousands of times too large. Attention takes a softmax over the
    sequence, so every training example longer than 256 tokens produced ~1e16 gradients
    and a NaN adapter. Subtracting the detached row maximum is exact (softmax is shift
    invariant) and keeps exp() in range.
    """
    if dtype is not None:
        input = input.to(dtype)
    dim = -1 if dim is None else dim
    exp = (input - input.amax(dim=dim, keepdim=True).detach()).exp()
    return exp / exp.sum(dim=dim, keepdim=True)


def use_composed_softmax() -> None:
    """Route attention softmax through ``composed_softmax`` for this process (XPU training)."""
    import torch

    torch.nn.functional.softmax = composed_softmax


def resource_snapshot(torch, device: str) -> dict:
    # The worker is launched as a file, so its sibling module needs no evolora import.
    from resources import device_resources

    return device_resources(torch, device)


def emit(event: str, **data) -> None:
    print(MARKER + json.dumps({"event": event, **data}), flush=True)


def pick_device():
    import torch

    # EVOLORA_DEVICE=cpu is the escape hatch when a GPU backend misbehaves.
    if os.environ.get("EVOLORA_DEVICE", "").lower() == "cpu":
        return "cpu", torch.float32
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", torch.bfloat16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", torch.float32  # Apple GPU; bf16 is not dependable on MPS
    return "cpu", torch.float32


def device_name(device: str) -> str:
    import torch

    if device == "xpu":
        return torch.xpu.get_device_name(0)
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return "Apple GPU (Metal)"
    return "CPU"


def peak_memory(device: str) -> int:
    import torch

    if device == "xpu":
        return int(torch.xpu.max_memory_allocated())
    if device == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device == "mps":
        return int(torch.mps.driver_allocated_memory())
    return 0


def chat_text(tokenizer, prompt: str, completion: str | None = None) -> str:
    """Same chat formatting for training, evaluation and chat, so they match."""
    messages = [{"role": "user", "content": prompt}]
    if completion is None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    messages.append({"role": "assistant", "content": completion})
    return tokenizer.apply_chat_template(messages, tokenize=False)


def training_rows(tokenizer, examples: list[dict], max_length: int) -> list[dict]:
    """Mask instructions and reject truncated answers instead of silently training them."""
    rows = []
    for index, example in enumerate(examples, 1):
        prompt = chat_text(tokenizer, example["prompt"])
        full = chat_text(tokenizer, example["prompt"], example["completion"]).rstrip()
        if not full.startswith(prompt):
            raise ValueError("Chat template does not preserve the generation prompt")
        if tokenizer.eos_token and not full.endswith(tokenizer.eos_token):
            full += tokenizer.eos_token
        tokens = tokenizer(full, add_special_tokens=False)["input_ids"]
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if tokens[:len(prompt_tokens)] != prompt_tokens:
            raise ValueError("Chat tokenization does not preserve the generation prompt")
        if len(tokens) > max_length:
            raise ValueError(f"Training example {index} needs {len(tokens)} tokens; "
                             f"sequence limit is {max_length}. Answer truncation refused")
        if len(tokens) <= len(prompt_tokens):
            raise ValueError(f"Training example {index} has no completion tokens")
        rows.append({"input_ids": tokens, "attention_mask": [1] * len(tokens),
                     "completion_mask": [0] * len(prompt_tokens)
                                        + [1] * (len(tokens) - len(prompt_tokens))})
    return rows


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

    device, dtype = ("cpu", torch.float32) if job.get("device") == "cpu" else pick_device()
    if device == "xpu":
        use_composed_softmax()
    started = time.perf_counter()
    model, tokenizer = load_model(job["base_model"], device, dtype)
    emit("loaded", device=device, device_name=device_name(device),
         load_s=round(time.perf_counter() - started, 2))

    hp = job["hyperparams"]
    from resources import GIB, microbatch_limit

    available = resource_snapshot(torch, device)
    effective_batch = hp["batch_size"] * hp["grad_accum"]
    batch = microbatch_limit(effective_batch, hp["batch_size"], available,
                            max(GIB // 2, GIB * job["max_length"] // 512))
    hp = {**hp, "batch_size": batch, "grad_accum": effective_batch // batch}
    emit("resources", resources=available, settings=hp,
         message=f"Live memory budget: microbatch {batch}, accumulation {hp['grad_accum']}")
    rows = training_rows(tokenizer, job["examples"], job["max_length"])
    output_dir = Path(job["output_dir"])

    class Progress(TrainerCallback):
        last_resources = 0.0

        def on_log(self, args, state, control, logs=None, **kwargs):
            validate_training_metrics(logs or {})
            if logs and "loss" in logs:
                emit("progress", step=state.global_step, total_steps=state.max_steps,
                     loss=round(float(logs["loss"]), 4))
            if time.monotonic() - self.last_resources >= 5:
                emit("resources", resources=resource_snapshot(torch, device), settings=hp)
                self.last_resources = time.monotonic()

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            validate_trainable_tensors(model, torch, gradients=True)

    config = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=hp["num_epochs"],
        per_device_train_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp["grad_accum"],
        learning_rate=hp["learning_rate"],
        optim="adamw_torch",
        use_cpu=device == "cpu",
        # Keep the frozen base in bf16, but avoid XPU bf16 autocast during adapter updates.
        bf16=device == "cuda" and dtype == torch.bfloat16,
        logging_nan_inf_filter=False,
        gradient_checkpointing=job.get("gradient_checkpointing", True),
        max_length=job["max_length"],
        dataset_text_field="text",
        completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True},
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
    if device == "xpu":
        # This worker uses a single device without a GradScaler; no unscale is needed.
        trainer.accelerator.clip_grad_norm_ = clip_xpu_gradients
    # Explicitly preserve fp32 adapters/AdamW state even when the frozen base is bf16.
    for parameter in trainer.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)
    # The default fused/foreach AdamW fails on the Arc 140V (Level Zero error 39).
    trainer.optimizer = torch.optim.AdamW(
        (p for p in trainer.model.parameters() if p.requires_grad),
        lr=hp["learning_rate"], foreach=False, fused=False,
    )
    train_started = time.perf_counter()
    result = trainer.train()
    validate_training_metrics({"loss": result.training_loss})
    validate_trainable_tensors(trainer.model, torch)
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    emit("done", adapter_dir=str(adapter_dir), train_loss=round(float(result.training_loss), 4),
         train_s=round(time.perf_counter() - train_started, 2), peak_bytes=peak_memory(device),
         steps=trainer.state.global_step, effective_hyperparams=hp, device=device)


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
    batch_cap = max(1, int(job.get("batch_size", 4)))
    from resources import GIB, microbatch_limit

    outputs: dict[str, str] = {}
    start = 0
    while start < len(prompts):
        available = resource_snapshot(torch, device)
        batch_size = microbatch_limit(batch_cap, batch_cap, available, GIB // 2)
        emit("resources", resources=available, settings={"inference_batch": batch_size})
        batch = prompts[start:start + batch_size]
        texts = [chat_text(tokenizer, p["prompt"]) for p in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        try:
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    max_new_tokens=job.get("max_new_tokens", 384),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except RuntimeError as exc:
            if batch_size <= 1 or not any(s in str(exc).lower() for s in OOM_SIGNS):
                raise
            batch_cap = max(1, batch_size // 2)
            del enc
            release_gpu_memory()
            emit("resources", resources=resource_snapshot(torch, device),
                 settings={"inference_batch": batch_cap},
                 message=f"Inference memory pressure; retrying batch with cap {batch_cap}")
            continue  # retry the same prompts; do not drop or shorten eval cases
        new_tokens = generated[:, enc["input_ids"].shape[1]:]
        for p, tokens in zip(batch, new_tokens, strict=True):
            outputs[p["sample_id"]] = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        emit("progress", done=len(outputs), total=len(prompts))
        start += len(batch)
        del generated, new_tokens, enc

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
        emit("error", error=message[:2000], oom=any(s in message.lower() for s in OOM_SIGNS),
             unstable="Numerically unstable training" in message)
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
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # best effort: process exit frees it anyway
        pass


if __name__ == "__main__":
    sys.exit(main())
