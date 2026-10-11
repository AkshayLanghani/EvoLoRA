"""EvoLoRA CLI — entry point for all commands."""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="evolora", help="EvoLoRA: auditable bounded self-improvement loop.")
console = Console()


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """Launch the TUI when 'evolora' is run with no subcommand."""
    if ctx.invoked_subcommand is None:
        tui()


@app.command()
def doctor() -> None:
    """Check environment and service connectivity."""
    from evolora.config import get_config

    cfg = get_config()
    table = Table(title="EvoLoRA Doctor", show_header=True)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    def row(name, ok, detail=""):
        status = "[green]OK[/green]" if ok else "[red]MISSING[/red]"
        table.add_row(name, status, detail)

    row("Python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0])
    row("EVOLORA_PROFILE", True, cfg.profile)
    if cfg.is_local:
        _doctor_local(cfg, row)
        console.print(table)
        return
    row("MINIMAX_API_KEY", cfg.minimax_available, "set" if cfg.minimax_available else "not set — mock mode")
    row(
        "DIGITAL_OCEAN_MODEL_ACCESS_KEY",
        cfg.digital_ocean_judge_available,
        "set" if cfg.digital_ocean_judge_available else "not set — heuristic judge",
    )
    row("MONGODB_URI", cfg.mongo_available, "set" if cfg.mongo_available else "not set — in-memory")
    row("TRAINING_BACKEND", True, cfg.training_backend)
    row("MODEL_RUNNER", True, cfg.model_runner)
    row("ARTIFACT_DIR", True, cfg.artifact_dir)

    console.print(table)

    if not cfg.minimax_available:
        console.print("[yellow]MiniMax not configured — heuristic planner will be used.[/yellow]")
    if not cfg.digital_ocean_judge_available:
        console.print("[yellow]DigitalOcean judge not configured — heuristic judge will be used.[/yellow]")
    if not cfg.mongo_available:
        console.print("[yellow]MongoDB not configured — using in-memory store.[/yellow]")


def _doctor_local(cfg, row) -> None:
    """Checks for the local profile: LLM server, models, storage and the training device."""
    from pathlib import Path

    import httpx

    from evolora.services import judge_spec, planner_spec

    if cfg.local_llm_managed:
        server = Path(cfg.llama_server_path) if cfg.llama_server_path else None
        row("LLAMA_SERVER_PATH", bool(server and server.is_file()), cfg.llama_server_path or "not set")
    for label, spec in (("Planner model", planner_spec(cfg)), ("Judge model", judge_spec(cfg))):
        if spec is None:
            row(label, True, "same as planner")
            continue
        row(label, bool(spec.path) and Path(spec.path).is_file(), f"{spec.alias}: {spec.path or 'no file'}")
    try:
        reachable = httpx.get(f"{cfg.local_llm_base_url.rstrip('/')}/models", timeout=2).status_code == 200
    except httpx.HTTPError:
        reachable = False
    started = "started on demand" if cfg.local_llm_managed else "start it yourself"
    row("Local LLM server", reachable, cfg.local_llm_base_url if reachable else f"not running ({started})")
    row("Run history (SQLite)", True, cfg.db_path)
    row("TRAINING_BACKEND", True, cfg.training_backend)
    _doctor_training_env(cfg, row)


_TRAINING_PROBE = """
import json, torch, transformers, peft, trl
dev = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
name = {"xpu": lambda: torch.xpu.get_device_name(0), "cuda": lambda: torch.cuda.get_device_name(0),
        "cpu": lambda: "CPU"}[dev]()
print(json.dumps({"torch": torch.__version__, "transformers": transformers.__version__,
                  "peft": peft.__version__, "trl": trl.__version__, "device": dev, "name": name}))
"""


def _doctor_training_env(cfg, row) -> None:
    """Probe the training interpreter in its own process (it has torch; we may not)."""
    import json
    import subprocess
    from pathlib import Path

    python = cfg.local_train_python
    if not Path(python).is_file():
        row("Training interpreter", False, f"not found: {python} (set LOCAL_TRAIN_PYTHON)")
        return
    row("Training interpreter", True, python)
    try:
        out = subprocess.run(
            [python, "-c", _TRAINING_PROBE], capture_output=True, text=True, timeout=120
        )
        info = json.loads(out.stdout.strip().splitlines()[-1])
    except (subprocess.SubprocessError, json.JSONDecodeError, IndexError):
        row("Training libraries", False, "torch/transformers/peft/trl not importable there")
        return
    row(
        "Training libraries",
        True,
        f"torch {info['torch']}, transformers {info['transformers']}, peft {info['peft']}, "
        f"trl {info['trl']}",
    )
    row("Training device", info["device"] != "cpu", f"{info['device']}: {info['name']}")


@app.command()
def demo(
    mock: bool = typer.Option(True, "--mock/--no-mock", help="Force mock backend"),
    iterations: int = typer.Option(3, help="Max iterations"),
) -> None:
    """Run a demo EvoLoRA loop (mock by default)."""
    asyncio.run(_demo(mock=mock, iterations=iterations))


async def _demo(mock: bool, iterations: int) -> None:
    import os

    from evolora.agent.planner import get_planner
    from evolora.config import get_config
    from evolora.demo.task import ADAPTIVE_EVAL_SET, LOCKED_EVAL_SET
    from evolora.evaluation.digitalocean_judge import get_judge
    from evolora.models.core import RunConfig
    from evolora.orchestration.orchestrator import Orchestrator
    from evolora.orchestration.retrain_advisor import get_retrain_advisor
    from evolora.training.backends import get_backend
    from evolora.training.runner import get_runner

    cfg = get_config()
    if mock:
        os.environ["TRAINING_BACKEND"] = "mock"

    run_config = RunConfig(
        max_iterations=iterations,
        target_score=cfg.target_score,
        training_backend="mock" if mock else cfg.training_backend,
        model_runner="mock" if mock else cfg.model_runner,
        base_model_id=cfg.base_model_id,
    )

    backend = get_backend("mock" if mock else cfg.training_backend)
    runner = get_runner("mock" if mock else cfg.model_runner)
    planner = get_planner(
        use_minimax=cfg.minimax_available and not mock,
        api_key=cfg.minimax_api_key,
        model=cfg.minimax_model,
        base_url=cfg.minimax_base_url,
    )
    judge = get_judge(
        api_key=cfg.digital_ocean_model_access_key,
        base_url=cfg.digitalocean_inference_base_url,
        model=cfg.digital_ocean_judge_model,
    )
    retrain_advisor = get_retrain_advisor(
        api_key=cfg.minimax_api_key if not mock else "",
        model=cfg.minimax_model,
        base_url=cfg.minimax_base_url,
    )

    orch = Orchestrator(
        config=run_config,
        eval_set=LOCKED_EVAL_SET,
        planner=planner,
        training_backend=backend,
        model_runner=runner,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=judge,
        retrain_advisor=retrain_advisor,
    )

    mode = "[yellow][MOCK][/yellow]" if mock else "[cyan][REAL][/cyan]"
    console.rule(f"EvoLoRA Demo {mode}")

    async for event in await orch.run():
        _print_event(event)

    console.rule("Done")


def _print_event(event) -> None:
    from evolora.models.events import EventKind

    labels = {
        EventKind.RUN_STARTED: "[bold green]START   [/]",
        EventKind.EVAL_SET_LOCKED: "[bold cyan]LOCK    [/]",
        EventKind.BASELINE_COMPLETE: "[bold blue]BASELINE[/]",
        EventKind.PLANNING_STARTED: "[bold yellow]PLAN    [/]",
        EventKind.PLAN_RECEIVED: "[bold yellow]PLAN OK [/]",
        EventKind.AGENT_FALLBACK_USED: "[bold red]FALLBACK[/]",
        EventKind.EVAL_APPROVAL_REQUIRED: "[bold yellow]APPROVE [/]",
        EventKind.TRAINING_STARTED: "[bold]TRAIN   [/]",
        EventKind.TRAINING_PROGRESS: None,  # skip progress spam in CLI
        EventKind.TRAINING_COMPLETE: "[bold green]TRAINED [/]",
        EventKind.EVAL_COMPLETE: "[bold blue]EVAL    [/]",
        EventKind.ADAPTIVE_COMPLETE: "[bold]ADAPTIVE[/]",
        EventKind.JUDGE_STARTED: "[bold cyan]JUDGE   [/]",
        EventKind.JUDGE_COMPLETE: "[bold cyan]JUDGED  [/]",
        EventKind.RETRAIN_DECISION_RECEIVED: "[bold yellow]DECIDE  [/]",
        EventKind.USER_APPROVAL_REQUIRED: "[bold yellow]APPROVE [/]",
        EventKind.USER_APPROVAL_RECEIVED: "[bold yellow]ANSWER  [/]",
        EventKind.ITERATION_COMPLETE: "[bold]ITER    [/]",
        EventKind.BEST_UPDATED: "[bold green]BEST    [/]",
        EventKind.STOP_CONDITION_MET: "[bold red]STOP    [/]",
        EventKind.RUN_COMPLETE: "[bold green]DONE    [/]",
        EventKind.RUN_FAILED: "[bold red]FAILED  [/]",
        EventKind.RUN_CANCELLED: "[bold red]CANCEL  [/]",
        EventKind.LOG: "        ",
    }
    label = labels.get(event.kind)
    if label is None:
        return
    console.print(f"{label} {event.message}")


@app.command("smoke-minimax")
def smoke_minimax() -> None:
    """Smoke-test MiniMax connectivity (requires MINIMAX_API_KEY)."""
    asyncio.run(_smoke_minimax())


async def _smoke_minimax() -> None:
    from evolora.agent.planner import _strip_think
    from evolora.config import get_config

    cfg = get_config()
    if not cfg.minimax_available:
        console.print("[red]MINIMAX_API_KEY not set.[/red]")
        raise typer.Exit(1)
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=cfg.minimax_api_key,
            base_url=cfg.minimax_base_url,
        )
        resp = await client.chat.completions.create(
            model=cfg.minimax_model,
            messages=[
                {"role": "user", "content": 'Reply with {"ok": true}'},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = _strip_think(resp.choices[0].message.content or "").strip()
        console.print(f"[green]MiniMax OK ({cfg.minimax_model}):[/green] {content}")
    except Exception as exc:
        console.print(f"[red]MiniMax FAILED:[/red] {exc}")
        raise typer.Exit(1)


@app.command("history")
def history(limit: int = typer.Option(10, help="Number of runs to show")) -> None:
    """Show recent runs from SQLite (EVOLORA_DB_PATH / local profile) or MongoDB."""
    from evolora.config import get_config
    from evolora.persistence.store import InMemoryRunStore, get_run_store

    store = get_run_store(get_config())
    if isinstance(store, InMemoryRunStore):
        console.print(
            "[yellow]No persistent store configured. Set EVOLORA_DB_PATH (or use the local "
            "profile) or MONGODB_URI.[/yellow]"
        )
        raise typer.Exit(1)
    records = asyncio.run(store.list_runs(limit=limit))
    table = Table(title="Recent runs")
    for column in ("Started", "Goal", "Status", "Baseline", "Best", "Stop reason"):
        table.add_column(column)
    for r in records:
        table.add_row(
            r.started_at.strftime("%Y-%m-%d %H:%M"),
            (r.config.goal or "(demo task)")[:48],
            r.status.value,
            f"{r.baseline_score:.3f}",
            f"{r.best_score:.3f}",
            r.stop_reason.value if r.stop_reason else "-",
        )
    console.print(table if records else "[dim]No runs yet.[/dim]")


@app.command("agent")
def agent(
    port: int = typer.Option(8765, help="Port on 127.0.0.1"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the browser"),
) -> None:
    """Run EvoLoRA fully on this machine and serve the web UI at http://127.0.0.1:PORT."""
    import os
    import threading
    import webbrowser

    # The agent is the local profile; set it before anything reads the config.
    os.environ.setdefault("EVOLORA_PROFILE", "local")
    # Keep run logs with the user's data (the working directory), not inside the app
    # folder, which the installer replaces on every update.
    os.environ.setdefault("EVOLORA_LOG_DIR", os.path.join(os.getcwd(), "logs"))
    from evolora.config import ConfigError, get_config

    try:
        cfg = get_config()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if not cfg.is_local:
        console.print("[red]EVOLORA_PROFILE is set to 'cloud'; the agent needs 'local'.[/red]")
        raise typer.Exit(1)

    import uvicorn

    url = f"http://127.0.0.1:{port}/"
    console.print(f"EvoLoRA agent on [bold]{url}[/bold]  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    # Loopback only: the agent runs training and LLM processes on this machine.
    uvicorn.run("evolora.web.app:app", host="127.0.0.1", port=port, log_level="warning")


@app.command("setup")
def setup(
    model: list[str] = typer.Option(
        None, "--model", "-m", help="Catalog key(s) to download; default: the configured planner"
    ),
    list_only: bool = typer.Option(False, "--list", help="Show the catalog and exit"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Download without asking"),
) -> None:
    """Download the local LLMs used by the local profile."""
    from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

    from evolora.config import get_config
    from evolora.local.models import CATALOG, DEFAULT_PLANNER, download, is_downloaded

    cfg = get_config()
    models_dir = cfg.local_models_dir
    table = Table(title=f"Local models (saved to {models_dir})")
    for column in ("Key", "Role", "Size", "Downloaded", "Note"):
        table.add_column(column)
    for m in CATALOG.values():
        table.add_row(
            m.key, m.role, f"{m.size_gb:.2f} GB", "yes" if is_downloaded(models_dir, m) else "no", m.note
        )
    console.print(table)
    if list_only:
        return

    keys = model or [cfg.local_llm_model if cfg.local_llm_model in CATALOG else DEFAULT_PLANNER]
    unknown = [k for k in keys if k not in CATALOG]
    if unknown:
        console.print(f"[red]Unknown model key(s): {', '.join(unknown)}[/red]")
        raise typer.Exit(1)
    todo = [CATALOG[k] for k in keys if not is_downloaded(models_dir, CATALOG[k])]
    if not todo:
        console.print("[green]Everything requested is already downloaded.[/green]")
        return
    total_gb = sum(m.size_gb for m in todo)
    sources = ", ".join(f"{m.repo}/{m.filename}" for m in todo)
    if not yes and not typer.confirm(f"Download {total_gb:.2f} GB from Hugging Face ({sources})?"):
        raise typer.Exit(1)

    with Progress(
        "[progress.description]{task.description}", BarColumn(), DownloadColumn(), TransferSpeedColumn()
    ) as progress:
        for m in todo:
            task = progress.add_task(m.key, total=None)
            download(m, models_dir, lambda done, total, t=task: progress.update(t, completed=done, total=total or None))
    console.print("[green]Done.[/green] Point the local profile at it with LOCAL_LLM_MODEL=<key>.")


@app.command("tui")
def tui() -> None:
    """Launch the Textual TUI (built by Codex)."""
    try:
        from evolora.tui.app import EvoLoRAApp  # type: ignore

        EvoLoRAApp().run()
    except ImportError:
        console.print("[yellow]TUI not yet built — Codex is on it.[/yellow]")


if __name__ == "__main__":
    app()
