"""An outer stream must release training before another GPU job starts."""

from evolora.demo.task import LOCKED_EVAL_SET
from evolora.models.core import RunConfig, RunStatus
from evolora.models.events import EventKind
from evolora.orchestration.orchestrator import Orchestrator
from evolora.persistence.store import InMemoryRunStore


class CancellableBackend:
    is_mock = True
    name = "cancellable"

    def __init__(self):
        self.closed = False

    async def health_check(self):
        return True

    async def train(self, *args, **kwargs):
        return self.stream()

    async def stream(self):
        try:
            yield {"done": False, "step": 1, "total_steps": 10, "loss": 1.0}
            yield {"done": False, "step": 2, "total_steps": 10, "loss": 1.0}
        finally:
            self.closed = True


async def test_closing_orchestration_closes_training_immediately(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = CancellableBackend()
    orch = Orchestrator(
        config=RunConfig(max_iterations=1), eval_set=LOCKED_EVAL_SET,
        training_backend=backend, run_store=InMemoryRunStore(),
    )
    stream = await orch.run()
    async for event in stream:
        if event.kind == EventKind.TRAINING_PROGRESS:
            break
    assert not backend.closed
    await stream.aclose()
    assert backend.closed


async def test_cancel_during_training_is_not_recorded_as_training_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = CancellableBackend()
    orch = Orchestrator(
        config=RunConfig(max_iterations=1), eval_set=LOCKED_EVAL_SET,
        training_backend=backend, run_store=InMemoryRunStore(),
    )
    events = []
    async for event in await orch.run():
        events.append(event)
        if event.kind == EventKind.TRAINING_PROGRESS:
            orch.cancel()
    assert backend.closed
    assert events[-1].kind == EventKind.RUN_CANCELLED
    assert orch._record.status == RunStatus.CANCELLED
    assert not orch._record.iterations
