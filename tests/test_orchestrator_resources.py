"""The orchestrator announces which resource each stage needs, in order."""

import pytest

from evolora.demo.task import LOCKED_EVAL_SET
from evolora.evaluation.digitalocean_judge import HeuristicJudge
from evolora.models.core import RunConfig
from evolora.models.events import EventKind
from evolora.orchestration.orchestrator import Orchestrator
from evolora.orchestration.retrain_advisor import HeuristicRetrainAdvisor
from evolora.persistence.store import InMemoryRunStore


class RecordingResources:
    def __init__(self, warning: str | None = None) -> None:
        self.calls: list[str] = []
        self._warning = warning

    async def acquire(self, resource):
        self.calls.append(resource)
        return self._warning


@pytest.fixture
def make_orchestrator(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # mock artifacts land under ./artifacts

    def _make(resources):
        return Orchestrator(
            config=RunConfig(max_iterations=1, training_sample_count=2),
            eval_set=LOCKED_EVAL_SET,
            run_store=InMemoryRunStore(),
            judge=HeuristicJudge(),
            retrain_advisor=HeuristicRetrainAdvisor(),
            resources=resources,
        )

    return _make


async def _run(orch):
    return [event async for event in await orch.run()]


async def test_stages_request_resources_in_order(make_orchestrator):
    resources = RecordingResources()
    await _run(make_orchestrator(resources))
    # baseline -> plan -> train -> judge -> advisor
    assert resources.calls == ["gpu", "planner", "gpu", "judge", "planner"]


async def test_resource_warnings_are_logged_not_fatal(make_orchestrator):
    events = await _run(make_orchestrator(RecordingResources("Local LLM unavailable")))
    logs = [e.message for e in events if e.kind == EventKind.LOG]
    assert "Local LLM unavailable" in logs
    assert events[-1].kind == EventKind.RUN_COMPLETE


class _ZeroRunner:
    is_mock = True

    async def run_batch(self, prompts, adapter_path=None):
        return {p["sample_id"]: "not json" for p in prompts}


class _AcceptingAdvisor(HeuristicRetrainAdvisor):
    async def decide(self, **kwargs):
        decision = await super().decide(**kwargs)
        return decision.model_copy(update={"retrain_recommended": False})


async def test_autonomous_run_ignores_acceptance_of_a_useless_adapter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    orch = Orchestrator(
        config=RunConfig(max_iterations=2, training_sample_count=2, target_score=0.85),
        eval_set=LOCKED_EVAL_SET,
        model_runner=_ZeroRunner(),
        run_store=InMemoryRunStore(),
        judge=HeuristicJudge(),
        retrain_advisor=_AcceptingAdvisor(),
    )
    events = await _run(orch)
    logs = [e.message for e in events if e.kind == EventKind.LOG]
    assert any(m.startswith("[guard]") for m in logs)
    assert len([e for e in events if e.kind == EventKind.ITERATION_COMPLETE]) == 2


async def test_no_resource_manager_is_a_no_op(make_orchestrator):
    events = await _run(make_orchestrator(None))
    assert events[-1].kind == EventKind.RUN_COMPLETE
