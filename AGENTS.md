# AGENTS.md

Shared instructions for every coding agent on this repo (Codex, Claude Code). Read this
file, then `docs/agents/BOARD.md` and `docs/agents/MESSAGES.md`, before doing anything.

## Project in one paragraph

EvoLoRA is a bounded self-improvement loop for LoRA fine-tuning (AI Engineer World's Fair
Hackathon 2026 finalist). A user states a goal; an LLM planner generates a locked
(SHA-256 hashed) eval set, synthesizes training data and picks LoRA hyperparameters
through three bounded tools (`create_evals`, `add_training_examples`,
`start_training_model`); Python validates everything, snaps hyperparameters onto fixed
choice sets, trains, scores against the locked evals, asks a judge and a retrain advisor,
and stops on one of nine conditions while keeping the best adapter. See `README.md` and
`docs/ARCHITECTURE.md`.

## Current goal: run 100% locally (branch `local-first`)

The owner wants the whole project to run on their laptop with no external services:

- Intel Core Ultra 7 256V (Lunar Lake), Intel Arc 140V iGPU (shares RAM, ~9 GB usable),
  Intel AI Boost NPU, 16 GB RAM, Windows 11, Python 3.11, `uv`.

The full plan, decisions and status are in `docs/LOCAL_PLAN.md`. Summary:

| Piece | Local replacement | Status |
| --- | --- | --- |
| MiniMax planner / eval gen / advisor | llama.cpp `llama-server` (OpenAI API) + Qwen3-4B GGUF | done (code), untested on hardware |
| DigitalOcean judge | same local server, optional second model | done (code) |
| MongoDB | SQLite (`evolora.db`) | done |
| Approval gates in web UI | `evolora agent` (local FastAPI) | done |
| Remote GPU VM training | Intel XPU training (Unsloth-XPU or PyTorch XPU + PEFT) | **not started** |
| LiveKit voice | whisper.cpp / Kokoro (optional, later) | not started |

## Layout you need

```
src/evolora/config.py              profiles: cloud (default) | local; ConfigError
src/evolora/services.py            builds planner/judge/advisor/resources per profile
src/evolora/local/supervisor.py    LlamaServer (start/swap/stop) + LocalResources
src/evolora/local/models.py        GGUF catalog + downloader (evolora setup)
src/evolora/orchestration/orchestrator.py   the loop; ResourceManager hooks
src/evolora/training/backends.py   TrainingBackend protocol, mock/remote/unsloth
src/evolora/persistence/sqlite_store.py     local run history
src/evolora/web/app.py             FastAPI API (hosted on Vercel or `evolora agent`)
src/virtual_machine_code/          train.py / evaluate.py / chat.py (GPU-side scripts)
public/                            website (vanilla HTML/CSS/JS, strict CSP: no inline JS/CSS)
tests/                             pytest; must stay green
```

## Commands

```powershell
uv sync
uv run pytest -q                 # all tests must pass before you hand off
uv run ruff check src api tests  # lint (tests/test_narrator.py, tests/test_voice.py have 4 old warnings)
uv run evolora doctor            # add EVOLORA_PROFILE=local to check the local stack
uv run evolora agent             # local web app at http://127.0.0.1:8765
```

## Rules for working alongside another agent

1. **Same working tree, same branch.** Never run `git checkout`, `git switch`, `git stash`,
   `git reset`, `git clean` or `git rebase`: the other agent may have uncommitted work.
   Do not commit or push unless the owner tells you to.
2. **Claim before you edit.** Put your name on a task in `docs/agents/BOARD.md` and list
   the files/directories you will touch. Do not edit files another agent has claimed;
   ask in `MESSAGES.md` instead.
3. **Talk through `docs/agents/MESSAGES.md`.** Append only, newest at the bottom, using
   the template there. Read it at the start of every task and before handing off.
4. **Hand off cleanly.** When you stop: tests green, BOARD updated, a message saying what
   changed, what was verified (with numbers), and what is left.
5. **Keep the cloud profile working.** Local work goes behind `EVOLORA_PROFILE=local`;
   mock mode must keep working with no keys, no GPU and no network.
6. **Report facts.** Benchmarks with exact versions and numbers; say plainly when
   something failed or was not tested.
7. Never commit `.env`, models (`models/`), adapters, `*.db`, or `workspace/`.
8. Match the surrounding code style (type hints, short docstrings that explain why,
   ruff line length 100).
