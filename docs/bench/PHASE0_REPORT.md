# Phase 0: Arc 140V feasibility measurements

Status: complete, 2026-10-10 (sections 0.1–0.6 by Codex; the integrated-system
measurements and the decision record by Claude). All measurements below were made on the owner's
Windows laptop. Scripts are in `scripts/bench/`; no project source or tests were edited
for these experiments. Downloads and benchmark environments are in ignored paths.

## Machine and method

- Intel Core Ultra 7 256V, Intel Arc 140V integrated GPU, 16 GB system RAM, Windows 11.
- Windows graphics driver `32.0.101.8626`; Level Zero device driver `1.14.37111`.
- PyTorch reports `8140 MB` XPU total memory (shared system memory),
  `Intel(R) oneAPI Unified Runtime over Level-Zero V2`.
- Python `3.11.9`, `uv`; isolated `.venv-bench` and `.venv-bench-unsloth`.
- XPU wheel bundles oneAPI runtime packages at `2026.1.0`; the oneAPI Base Toolkit was
  not separately installed (`C:\Program Files (x86)\Intel\oneAPI` absent).
- Primary training versions: `torch 2.14.1+xpu`, `transformers 4.57.3`,
  `peft 0.21.2`, `trl 0.24.0`, `accelerate 1.15.0`, `datasets 5.0.1`,
  `bitsandbytes 0.50.2`. Exact install recipe: `scripts/bench/requirements-xpu.txt`.
- Inference build: llama.cpp `b11552`, commit `23b0202a1`, 0.6.0-dev,
  Windows SYCL and Vulkan binaries.

The SFT workload is 40 repeats of four short prompt/JSON-answer pairs, LoRA rank 8,
alpha 16, q/k/v/o targets, learning rate 2e-4, batch 1, maximum 128 tokens,
bf16, 10 optimizer steps. `scripts/bench/train_sft.py` synchronizes XPU at every step.
The steady-step figure excludes steps 1–2. This is a *feasibility microbenchmark*;
longer sequences and real training data need separate measurement.

## 0.1 Intel GPU stack

**Pass.** `torch.xpu.is_available()` returned `True`; the device is
`Intel(R) Arc(TM) 140V GPU (8GB)`. Ten synchronized 2048×2048 bf16 matrix
multiplications averaged **0.657 ms** per multiplication. Output dtype was bf16.
`scripts/bench/gpu_probe.py` reproduces this check.

## 0.2 PEFT/TRL Qwen2.5-0.5B on Windows

**Pass**, after disabling fused and foreach AdamW paths.

| Configuration | 10-step train time | Steps 3–10 | Peak XPU allocated | Final step loss |
| --- | ---: | ---: | ---: | ---: |
| Gradient checkpointing off | 6.21 s | 0.254 s/step | 1,128,224,768 B (1.051 GiB) | 3.0156 |
| Gradient checkpointing on | 5.54 s | 0.237 s/step | 1,094,095,872 B (1.019 GiB) | 3.0393 |

The two steady speeds are similar at this short sequence length. The first step took
3.2–3.8 s for kernel warmup. Raw benchmark output is under ignored `models/bench-05b*.txt`.

Failures and fixes:

1. `trl 1.15.0` with `transformers 5.19.0` failed before step 1 in TRL's fused LM head
   with `torch.OutOfMemoryError` requesting an implausible `2526058320.58 GiB`.
2. `trl 0.24.0` with the default fused Adam failed at `optimizer.step()` with
   `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` (Level Zero error 39).
3. `optim="adamw_torch"` alone still failed in `_multi_tensor_adam` at step 2.
4. Explicit `torch.optim.AdamW(trainable_parameters, lr=2e-4,
   foreach=False, fused=False)` completed all 10 steps.

## 0.3 Unsloth Intel on Windows

Pending installation and benchmark. A plain `uv pip install unsloth --torch-backend=xpu`
resolved the obsolete `unsloth 2024.8` package without dependencies; its import failed
with `ModuleNotFoundError: No module named 'packaging'`. The official Intel source extra
is being attempted separately.

## 0.4 WSL2

**Not run.** `wsl --list --verbose` reports that WSL is not installed. Installing it
requires administrator rights and a reboot; no WSL environment was created for this
benchmark.

## 0.5 Larger training models

Qwen2.5-1.5B-Instruct bf16 LoRA **fits and trains** using the same PEFT/TRL stack,
gradient checkpointing, and explicit non-fused AdamW: 10 steps in 12.06 s including
warmup; steps 3–10 averaged **0.704 s/step** (one 2.16 s outlier); peak XPU allocated
**3,200,524,800 B (2.98 GiB)** and reserved **3,401,580,544 B**. Final step loss
was **2.9941**; mean training loss **3.6534**.

Phi-3-mini 4-bit QLoRA via bitsandbytes XPU: pending.

## 0.6 llama.cpp inference and tool calls

The 4B model is `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (2,497,281,120 bytes) from
`unsloth/Qwen3-4B-Instruct-2507-GGUF`. SYCL b11552 used `-ngl 99 -c 4096 --jinja`
on `127.0.0.1:8090`. Tool trials used the exact `TOOLS` list from
`src/evolora/agent/tools.py`, five varied task goals, and the planner's forced-call
temperature of 0.9. `jsonschema` validated arguments against the `create_evals`
parameters schema. Raw 4B SYCL records: `docs/bench/4b-sycl.json`.

| Build/model | Forced-form request: schema-valid `create_evals` | Generated speed | Notes |
| --- | ---: | ---: | --- |
| SYCL / 4B | 20/20 | mean 21.55 tok/s; median 21.80; range 15.15–27.35 | `tool_choice` object was ignored; see below |
| Vulkan / 4B | pending | pending | |
| SYCL / 8B | pending | pending | |
| Vulkan / 8B | pending | pending | |

**Important compatibility finding:** llama.cpp b11552 logged
`Wrong type supplied for parameter 'tool_choice', using default value: type must be
string, but is object`. Thus the object-form forced choice used by EvoLoRA was **not
honored**. All 20 calls were valid because the prompt explicitly requested
`create_evals`; this does not establish forced-tool reliability. A `"required"` request
with only `create_evals` offered will be tested separately.

The first cold 4B SYCL server load took about 61 s (from server log timestamps).
Subsequent startup and shutdown are pending the timed handoff run.

## 0.7 Memory handoff

Pending. `scripts/bench/handoff.py` will measure start → stop → short 0.5B train →
stop → warm restart, including `torch.xpu.mem_get_info()` before and after each stage.

## Measurements from the integrated system (Claude, after Codex ran out of tokens)

- **Real EvoLoRA run on the Arc 140V** (Qwen3-4B planner via llama-server SYCL,
  Qwen2.5-0.5B trained, 6 examples, 1 iteration): 8 min 9 s end to end. Eval
  generation 40 s, baseline 38 s, planning 4 min 48 s (the slowest stage, ~21 tok/s),
  training 48 s including model load (12 steps, loss 2.58 → 2.01), eval 33 s, judge +
  advisor 31 s.
- **Model loads:** first-ever XPU model load 94 s (one-time kernel setup); later loads
  ~12 s. llama-server warm start ~14 s after the first cold load (61 s).
- **Forced tool choice:** b11552 rejects the object form, so the local planner sends
  `tool_choice="required"` with only `create_evals` offered.
- **Memory handoff failure seen in practice:** restarting llama-server right after
  training failed with Level Zero error 39 (out of device memory) at `--ctx-size 16384`.
  Fixes: default context 8192, up to three start attempts with back-off, the last at
  half the context, and the worker empties the XPU cache before exiting.
- **Clean-account install:** `EvoLoRA-Setup.cmd` + in-app "Install everything" produced
  a working stack with `torch 2.14.1+xpu` (via `uv pip install --torch-backend xpu`),
  transformers 4.57.3, peft 0.21.2, trl 0.24.0; 7.0 GB on disk.
- Not measured: Vulkan vs SYCL tok/s (SYCL works and is the default), Unsloth on Intel
  (its install resolved an obsolete package), Phi-3-mini 4-bit, WSL2.

## Decision record

| Decision | Choice | Why |
| --- | --- | --- |
| OS path | Windows native | Works end to end; WSL2 needs admin + reboot and wasn't installed |
| Training stack | PyTorch XPU + PEFT + TRL 0.24 (not Unsloth) | Measured working; Unsloth's Intel install failed; TRL 1.x failed |
| Optimizer | `AdamW(foreach=False, fused=False)` | Fused/foreach paths fail with Level Zero error 39 |
| llama.cpp build | SYCL b11552 on Intel; CUDA 12.4 on NVIDIA; Vulkan otherwise | SYCL measured 21–27 tok/s on 4B |
| Planner model | Qwen3-4B-Instruct-2507 Q4_K_M | 20/20 schema-valid `create_evals`, ~21 tok/s, 2.3 GB |
| Trainable base models | Qwen2.5-1.5B-Instruct (default), Qwen2.5-0.5B-Instruct | 1.5B peaks ~3 GB; Llama 3.1 8B and Phi-3 (bf16) don't fit |
| `LOCAL_LLM_CTX` | 8192 | 16384 overflowed shared memory after training |
| `LOCAL_LLM_GPU_LAYERS` | 99 (all) | Model fits fully on the iGPU |
| `LOCAL_LLM_MAX_TOKENS` | 4096 | Fits within 8k context with the planner prompt |
| `LOCAL_LLM_EXAMPLE_BATCH_SIZE` | 8 | Smaller tool payloads for a 4B model |
| Install | `EvoLoRA-Setup.cmd`, then "Install everything" in the app | See README "Run Locally" |
