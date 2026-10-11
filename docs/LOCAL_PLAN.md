# Local-first EvoLoRA: plan and status

Goal: every function of EvoLoRA runs on the owner's laptop with no external service.

**Machine:** Intel Core Ultra 7 256V (Lunar Lake, 8 cores), Intel Arc 140V iGPU (no
dedicated VRAM; Windows lets it use ~57% of RAM, about 9 GB), Intel AI Boost NPU, 16 GB
RAM, Windows 11, 600 GB free disk.

## What changes

| Piece | Cloud profile | Local profile |
| --- | --- | --- |
| Planner, eval generation, training data, retrain advisor | MiniMax M2.7 | `llama-server` (llama.cpp, OpenAI-compatible) with Qwen3-4B-Instruct Q4 (2.33 GB) or Qwen3-8B Q4 (4.68 GB) |
| LLM judge | Llama 3.3 70B on DigitalOcean | same local server; optional second model (Phi-4-mini, 2.32 GB) so the planner isn't grading itself |
| Training | Unsloth on a remote NVIDIA VM | Arc 140V via Unsloth-XPU or PyTorch XPU + PEFT/TRL (Phase 0 decides) |
| Trainable base models | Phi-3-mini, Llama 3.1 8B | Qwen2.5-1.5B / Qwen3-1.7B; Phi-3-mini only if 4-bit QLoRA works on XPU. Llama 3.1 8B dropped |
| History | MongoDB Atlas | SQLite file |
| Approvals | TUI | TUI and the local web UI (`evolora agent`) |
| Voice | LiveKit (Deepgram, Cartesia) | whisper.cpp / faster-whisper + Kokoro-82M (optional, later) |
| Dropped | | remote VM SSH backend, Gemini judge, Vercel (for local mode) |

Memory rule: the local LLM and training never share memory. `LocalResources` stops
`llama-server` before GPU stages and restarts it (with the right model) for LLM stages.

## Phases

### Phase 0: feasibility (no project code) — owner: Codex

Measure on this laptop, with exact versions:

| # | Experiment | Measure | Pass bar |
| --- | --- | --- | --- |
| 0.1 | Intel GPU driver + oneAPI + PyTorch XPU wheel | `torch.xpu.is_available()`, device name, a bf16 matmul | works |
| 0.2 | LoRA SFT on Qwen2.5-0.5B-Instruct with PEFT/TRL on XPU, Windows | s/step, peak memory, final loss | finishes, < 2 s/step |
| 0.3 | Same with Unsloth (Intel install) on Windows | same | compare with 0.2 |
| 0.4 | Faster of 0.2/0.3 under WSL2 (if Level Zero works there) | same | compare with Windows |
| 0.5 | Qwen2.5-1.5B bf16 LoRA; then Phi-3-mini 4-bit (bitsandbytes XPU) | fits? s/step, memory | 1.5B must fit |
| 0.6 | llama-server (SYCL build and Vulkan build) with Qwen3-4B / Qwen3-8B | tok/s; tool calls incl. forced `tool_choice` naming `create_evals`; JSON validity over 20 trials | 4B >= 20 tok/s, >= 90% valid |
| 0.7 | Handoff: start server -> stop -> train -> stop -> start | swap time, leftover memory | < 30 s, no leak |

Use the real tool schemas from `src/evolora/agent/tools.py` (`TOOLS`) for 0.6.
Output: `docs/bench/PHASE0_REPORT.md` with a decision record (Windows or WSL2; Unsloth
or PEFT; SYCL or Vulkan; default planner; trainable base models; recommended
`LOCAL_LLM_*` settings), plus the scripts in `scripts/bench/`.

### Phase 1: local brain — owner: Claude — code done

- `EVOLORA_PROFILE=local` routes planner, judge, advisor and narrator to
  `LOCAL_LLM_BASE_URL` (must be loopback) and blocks MongoDB, LiveKit and remote training.
- `LlamaServer` / `LocalResources` start, swap and stop `llama-server`.
- Planner timeouts, token limits and batch sizes are configurable for slow local models.
- Untested against a real llama-server until Phase 0 provides one.

### Phase 2: local training — owner: Claude — code done, end-to-end run in progress

- `src/evolora/local/worker.py`: standalone worker (`train` / `generate`) run by the
  training interpreter, one job per process. Recipe from Phase 0: TRL 0.24 SFT + PEFT
  LoRA (q/k/v/o), bf16, eager attention, gradient checkpointing,
  `AdamW(foreach=False, fused=False)`, tqdm off. Picks XPU, then CUDA, then CPU.
- `src/evolora/training/local_backend.py`: `LocalTrainingBackend` (progress events, batch
  capped with gradient accumulation, one OOM retry at batch 1 / half length, adapters
  archived with `evolora.json` metadata) and `LocalModelRunner` (batched greedy
  generation for baseline/eval scoring; scoring stays in the main process).
- Web: `/api/adapters`, `/api/chat` and a chat panel; runs one at a time locally.
- Planner uses `tool_choice="required"` locally (llama.cpp ignores the named form).
- Missing training interpreter falls back to the mock trainer with a notice.

### Phase 3: local app — owner: Claude — mostly done

- `evolora agent` (loopback-only FastAPI + site), approvals in the web UI, SQLite history,
  `evolora setup` (GGUF downloads), local checks in `evolora doctor`. Adapter chat lands
  with Phase 2.

### Phase 4: hardening — unassigned

Failure drills (kill llama-server mid-plan, forced OOM, browser closed mid-run, eval
tampering, network unplugged), timing table, `pytest -m hardware` suite, pinned versions.

### Phase 5: "Connect my computer" (owner's chosen next route, after local is finished)

The Vercel site becomes the front door for real, fully local runs on any visitor's
machine. No hosted AI APIs: all planning, judging and training happen on the visitor's
computer.

- The site detects whether an EvoLoRA agent is running on the visitor's machine; if not,
  it offers a download/installer for their OS.
- A secure pairing between the hosted page and the local agent (one-time code, origin
  allow-list, loopback only, Private Network Access headers), then the site drives the
  agent: runs, approvals, history, chat.
- Hardware detection on the visitor's machine picks the training path (Intel XPU, NVIDIA,
  Apple MLX, CPU) and suggests models that fit.
- Design details to be decided once local mode is complete.

Also optional: local voice; judge on the NPU via OpenVINO.

## Configuration (local profile)

See `.env.example`. Key settings: `EVOLORA_PROFILE=local`, `LLAMA_SERVER_PATH`,
`LOCAL_LLM_MODEL` (catalog key) or `LOCAL_LLM_MODEL_PATH`, `LOCAL_JUDGE_MODEL`,
`LOCAL_LLM_CTX`, `LOCAL_LLM_GPU_LAYERS`, `LOCAL_LLM_MANAGED`, `EVOLORA_DB_PATH`.
