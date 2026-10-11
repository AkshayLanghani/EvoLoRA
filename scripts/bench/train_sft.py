"""Small, reproducible LoRA SFT workload for the Phase 0 XPU comparisons."""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

EXAMPLES = [
    ("Return JSON with the sentiment of: I love this tool.", '{"sentiment":"positive"}'),
    ("Return JSON with the sentiment of: This is unusable.", '{"sentiment":"negative"}'),
    ("Return JSON with the sentiment of: The package arrived.", '{"sentiment":"neutral"}'),
    ("Return JSON with the sentiment of: What a wonderful result!", '{"sentiment":"positive"}'),
]


class StepTimer(TrainerCallback):
    """Synchronize at step boundaries so GPU work is included in each timing."""

    def __init__(self) -> None:
        self.last = 0.0
        self.steps: list[float] = []

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        torch.xpu.synchronize()
        self.last = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs) -> None:
        torch.xpu.synchronize()
        now = time.perf_counter()
        self.steps.append(now - self.last)
        self.last = now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--backend", choices=("peft", "unsloth"), default="peft")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    if not torch.xpu.is_available():
        raise RuntimeError("PyTorch XPU is unavailable")

    started = time.perf_counter()
    if args.backend == "unsloth":
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=128,
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
        )
        peft_config = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="eager"
        )
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
    loaded_s = time.perf_counter() - started
    dataset = Dataset.from_list(
        [{"text": f"<|im_start|>user\n{prompt}<|im_end|>\n"
                  f"<|im_start|>assistant\n{answer}<|im_end|>"}
         for prompt, answer in EXAMPLES] * max(4, args.steps)
    )
    output_dir = Path("models/bench-output")
    config = SFTConfig(
        output_dir=str(output_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        max_length=128,
        dataset_text_field="text",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    timer = StepTimer()
    trainer.add_callback(timer)
    trainer.optimizer = torch.optim.AdamW(
        (parameter for parameter in trainer.model.parameters() if parameter.requires_grad),
        lr=2e-4,
        foreach=False,
        fused=False,
    )
    torch.xpu.reset_peak_memory_stats()
    train_started = time.perf_counter()
    try:
        result = trainer.train()
    except Exception as exc:
        print("BENCH_ERROR=" + json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "peak_allocated_bytes": torch.xpu.max_memory_allocated(),
            "peak_reserved_bytes": torch.xpu.max_memory_reserved(),
            "elapsed_s": time.perf_counter() - train_started,
            "log_history": trainer.state.log_history,
            "step_times_s": timer.steps,
        }), flush=True)
        raise
    torch.xpu.synchronize()
    train_s = time.perf_counter() - train_started
    print("BENCH_RESULT=" + json.dumps({
        "model": args.model,
        "backend": args.backend,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "model_load_s": loaded_s,
        "train_s": train_s,
        "s_per_step": train_s / args.steps,
        "steady_s_per_step": sum(timer.steps[2:]) / max(1, len(timer.steps[2:])),
        "step_times_s": timer.steps,
        "peak_allocated_bytes": torch.xpu.max_memory_allocated(),
        "peak_reserved_bytes": torch.xpu.max_memory_reserved(),
        "final_loss": result.training_loss,
        "log_history": trainer.state.log_history,
    }))


if __name__ == "__main__":
    main()
