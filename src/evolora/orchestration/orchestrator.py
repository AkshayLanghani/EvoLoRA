"""EvoLoRA orchestrator — bounded self-improvement state machine."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import UTC, datetime
from typing import Literal, Protocol

from evolora.agent.planner import HeuristicPlanner, MiniMaxPlanner
from evolora.evaluation.digitalocean_judge import CandidateJudge
from evolora.evaluation.evaluator import GenericEvaluator, ObjectiveEvaluator
from evolora.evaluation.locked import LockedEvalSet
from evolora.models.core import (
    AgentPlan,
    ArtifactMeta,
    EvalResult,
    EvalSample,
    IterationResult,
    RetrainDecision,
    RunConfig,
    RunRecord,
    RunStatus,
    StopReason,
)
from evolora.models.events import Event, EventKind
from evolora.observability.run_logger import RunLogger
from evolora.orchestration.retrain_advisor import RetrainAdvisor
from evolora.persistence.artifacts import ArtifactStore, LocalArtifactStore
from evolora.persistence.store import RunStore, get_run_store
from evolora.training.backends import MockTrainingBackend, TrainingBackend
from evolora.training.remote_config import (
    build_baseline_config_payload,
    build_training_config_payload,
)
from evolora.training.runner import MockModelRunner, ModelRunner


def _now():
    return datetime.now(UTC)


# If the base model already scores at/above this on the generated evals, the evals are too
# easy for it — regenerate a harder set once so there's room (and meaning) to improve.
HARD_EVAL_THRESHOLD = 0.85

# Below this judge rating, an advisor's "accept" on an adapter that didn't beat the
# baseline is treated as a mistake in autonomous runs.
LOW_JUDGE_RATING = 0.5

Resource = Literal["planner", "judge", "gpu"]


class ResourceManager(Protocol):
    """Hands the machine's memory to whichever stage runs next.

    On a laptop the local LLM and model training can't share memory, so the orchestrator
    announces what it needs before each stage. Returns a warning to log, or None.
    """

    async def acquire(self, resource: Resource) -> str | None: ...


def _label(component, default: str) -> str:
    return getattr(component, "label", default)


class Orchestrator:
    """Drives the EvoLoRA loop: baseline → plan → train → eval → repeat."""

    def __init__(
        self,
        config: RunConfig,
        eval_set: LockedEvalSet,
        *,
        planner: MiniMaxPlanner | HeuristicPlanner | None = None,
        training_backend: TrainingBackend | None = None,
        model_runner: ModelRunner | None = None,
        run_store: RunStore | None = None,
        artifact_store: ArtifactStore | None = None,
        adaptive_eval_set: LockedEvalSet | None = None,
        judge: CandidateJudge | None = None,
        retrain_advisor: RetrainAdvisor | None = None,
        llm_judge=None,
        run_logger: RunLogger | None = None,
        resources: ResourceManager | None = None,
        notices: list[str] | None = None,
    ) -> None:
        self._config = config
        self._resources = resources
        self._notices = list(notices or [])  # logged right after the run starts
        self._eval_set = eval_set
        self._adaptive_set = adaptive_eval_set
        self._planner = planner or HeuristicPlanner()
        self._backend = training_backend or MockTrainingBackend()
        self._runner = model_runner or MockModelRunner()
        self._judge = judge
        self._retrain_advisor = retrain_advisor
        self._llm_judge = llm_judge
        self._store = run_store or get_run_store()
        self._artifacts = artifact_store or LocalArtifactStore("./artifacts")
        self._evaluator = ObjectiveEvaluator()
        self._cancelled = False
        self._approval_future: asyncio.Future[bool] | None = None
        self._record = RunRecord(config=config)
        self._run_logger = run_logger or RunLogger(self._record.run_id)
        self._accumulated_examples: list[dict] = []
        self._keep_training = False
        self._last_baseline_score = 0.0
        self._baseline_ok = True
        self._baseline_results: list[EvalResult] = []

    def cancel(self) -> None:
        self._cancelled = True
        self.submit_retrain_approval(False)

    def submit_retrain_approval(self, approved: bool) -> None:
        if self._approval_future is not None and not self._approval_future.done():
            self._approval_future.set_result(approved)

    async def run(self) -> AsyncIterator[Event]:
        return self._run()

    async def _acquire(self, resource: Resource) -> str | None:
        if self._resources is None:
            return None
        return await self._resources.acquire(resource)

    async def _run(self) -> AsyncIterator[Event]:
        rec = self._record
        rid = rec.run_id

        async def emit(kind: EventKind, msg: str = "", **data) -> Event:
            ev = Event(kind=kind, run_id=rid, iteration=rec.current_iteration(), message=msg, data=data)
            self._run_logger.log_event(ev)
            return ev

        # --- PREPARING ---
        rec.status = RunStatus.PREPARING
        yield await emit(EventKind.RUN_STARTED, "EvoLoRA run started", mock=self._backend.is_mock)
        for notice in self._notices:
            yield await emit(EventKind.LOG, f"[warn] {notice}")
        await self._store.save(rec)

        if rec.config.goal and not self._backend.is_mock and not isinstance(self._planner, MiniMaxPlanner):
            rec.error = "Goal-specific planner unavailable; refusing to train against demo evaluations"
            yield await emit(EventKind.LOG, rec.error)
            yield await self._finish(rec, RunStatus.FAILED, StopReason.BACKEND_UNAVAILABLE, emit)
            return

        # --- MINIMAX-GENERATED EVALS (goal-driven) ---
        # When the user supplied a goal and the agent can generate evals, MiniMax
        # produces a goal-specific objective eval set; we score it with the generic
        # evaluator and skip the demo adaptive set (which is customer-spending).
        generated_eval_records: list[dict[str, str]] | None = None
        if rec.config.goal and isinstance(self._planner, MiniMaxPlanner):
            if warning := await self._acquire("planner"):
                yield await emit(EventKind.LOG, warning)
            yield await emit(
                EventKind.STATUS_CHANGED,
                f"{_label(self._planner, 'MiniMax')} generating evals for goal: {rec.config.goal[:60]}",
            )
            # Evals are made up on the spot from the goal — no canned set. Retry transient
            # failures rather than silently dropping to the (off-topic) fallback set.
            generated: list[dict] | None = None
            for attempt in range(1, 4):
                try:
                    generated = await self._planner.generate_evals(rec.config.goal)
                    if generated:
                        break
                    yield await emit(EventKind.LOG, f"Eval generation returned nothing (try {attempt}/3), retrying…")
                except Exception as exc:
                    yield await emit(EventKind.LOG, f"Eval generation error (try {attempt}/3): {exc}")
            if generated:
                samples = [
                    EvalSample(sample_id=f"gen-{i + 1:03d}", prompt=g["prompt"], expected=g["expected"])
                    for i, g in enumerate(generated)
                ]
                self._eval_set = LockedEvalSet(samples)
                self._evaluator = GenericEvaluator()
                self._adaptive_set = None
                generated_eval_records = self._eval_records_for_approval(samples)
                yield await emit(
                    EventKind.LOG,
                    f"{_label(self._planner, 'MiniMax')} generated {len(samples)} goal-specific eval examples",
                )
            else:
                if not self._backend.is_mock:
                    rec.error = "Goal-specific eval generation failed after 3 attempts; training was not started"
                    yield await emit(EventKind.LOG, rec.error)
                    yield await self._finish(rec, RunStatus.FAILED, StopReason.BACKEND_UNAVAILABLE, emit)
                    return
                yield await emit(
                    EventKind.LOG,
                    f"[warn] {_label(self._planner, 'MiniMax')} could not generate evals after 3 tries "
                    "— using fallback eval set",
                )

        # In autonomous mode (require_retrain_approval off) the generated eval set is
        # auto-approved — no gate.
        if generated_eval_records and rec.config.require_retrain_approval:
            self._approval_future = asyncio.get_running_loop().create_future()
            yield await emit(
                EventKind.EVAL_APPROVAL_REQUIRED,
                f"Approve {len(generated_eval_records)} generated eval examples?",
                evals=generated_eval_records,
            )
            approved = await self._approval_future
            self._approval_future = None
            yield await emit(
                EventKind.USER_APPROVAL_RECEIVED,
                "User approved generated eval set" if approved else "User declined generated eval set",
                approved=approved,
                approval_type="evals",
            )
            if self._cancelled or not approved:
                self._cancelled = True
                yield await self._finish(rec, RunStatus.CANCELLED, StopReason.CANCELLED, emit)
                return

        # --- LOCK EVAL SET ---
        rec.status = RunStatus.LOCKING_EVAL
        yield await emit(EventKind.EVAL_SET_LOCKED, f"Eval set locked (hash={self._eval_set.hash[:12]}…)", hash=self._eval_set.hash, size=len(self._eval_set))
        await self._store.save(rec)
        rec.eval_set_hash = self._eval_set.hash

        # --- BASELINE ---
        rec.status = RunStatus.BASELINE
        yield await emit(EventKind.STATUS_CHANGED, "Running baseline evaluation")
        async for ev in self._run_baseline(rec, rid, emit):
            yield ev
        if not self._baseline_ok:
            yield await self._finish(rec, RunStatus.FAILED, StopReason.BACKEND_UNAVAILABLE, emit)
            return
        baseline_score = self._last_baseline_score
        rec.baseline_score = baseline_score
        rec.best_score = baseline_score
        yield await emit(EventKind.BASELINE_COMPLETE, f"Baseline score: {baseline_score:.3f}",
                         score=baseline_score, sample_results=[r.model_dump() for r in self._baseline_results])
        if snapshot := getattr(self._runner, "last_resources", None):
            yield await emit(EventKind.RESOURCE_STATUS, "Baseline resource availability",
                             resources=snapshot["resources"], settings=snapshot.get("settings", {}))
        await self._store.save(rec)

        # --- ADAPTIVE EVAL DIFFICULTY ---
        # If the base model already aces the evals, they're too easy for a capable model —
        # regenerate a harder set once and re-baseline so there's room (and meaning) to improve.
        if (
            baseline_score >= HARD_EVAL_THRESHOLD
            and rec.config.goal
            and isinstance(self._planner, MiniMaxPlanner)
        ):
            yield await emit(
                EventKind.LOG,
                f"Base model scored {baseline_score:.2f} (>= {HARD_EVAL_THRESHOLD:.2f}) — evals too "
                "easy; generating a harder set.",
            )
            if warning := await self._acquire("planner"):
                yield await emit(EventKind.LOG, warning)
            try:
                harder = await self._planner.generate_evals(rec.config.goal, difficulty="hard")
            except Exception:
                harder = None
            if harder:
                samples = [
                    EvalSample(sample_id=f"hard-{i + 1:03d}", prompt=g["prompt"], expected=g["expected"])
                    for i, g in enumerate(harder)
                ]
                self._eval_set = LockedEvalSet(samples)
                rec.eval_set_hash = self._eval_set.hash
                yield await emit(
                    EventKind.EVAL_SET_LOCKED,
                    f"Harder eval set locked ({len(samples)} examples, hash={self._eval_set.hash[:12]}…)",
                    hash=self._eval_set.hash,
                    size=len(self._eval_set),
                )
                rec.status = RunStatus.BASELINE
                async for ev in self._run_baseline(rec, rid, emit):
                    yield ev
                if not self._baseline_ok:
                    yield await self._finish(rec, RunStatus.FAILED, StopReason.BACKEND_UNAVAILABLE, emit)
                    return
                baseline_score = self._last_baseline_score
                rec.baseline_score = baseline_score
                rec.best_score = baseline_score
                yield await emit(
                    EventKind.BASELINE_COMPLETE,
                    f"Baseline on harder evals: {baseline_score:.3f}",
                    score=baseline_score,
                )
                await self._store.save(rec)

        # --- MAIN LOOP ---
        for iteration in range(1, rec.config.max_iterations + 1):
            if self._cancelled:
                yield await self._finish(rec, RunStatus.CANCELLED, StopReason.CANCELLED, emit)
                return

            # --- PLAN ---
            rec.status = RunStatus.PLANNING
            yield await emit(EventKind.PLANNING_STARTED, f"Iteration {iteration}: requesting plan", iteration=iteration)

            prev_score = rec.iterations[-1].score if rec.iterations else baseline_score
            failures = self._get_failures(rec)
            if warning := await self._acquire("planner"):
                yield await emit(EventKind.LOG, warning)
            plan, fallback = await self._plan(
                iteration,
                baseline_score,
                prev_score,
                failures,
                rec.config.training_sample_count,
                rec.config.goal,
            )

            if fallback:
                fallback_reason = str(getattr(self._planner, "last_error", "")).strip()
                message = f"{_label(self._planner, 'MiniMax')} unavailable — heuristic plan used"
                if fallback_reason:
                    message = f"{message}: {fallback_reason[:180]}"
                yield await emit(
                    EventKind.AGENT_FALLBACK_USED,
                    message,
                    reason=fallback_reason,
                )
                if rec.config.goal and not self._backend.is_mock:
                    rec.error = "Goal-specific planning failed; refusing off-topic heuristic training data"
                    yield await emit(EventKind.LOG, rec.error)
                    yield await self._finish(rec, RunStatus.FAILED, StopReason.BACKEND_UNAVAILABLE, emit)
                    return
            yield await emit(EventKind.PLAN_RECEIVED, "Plan received", rationale=plan.rationale[:200], focus_areas=plan.focus_areas)

            # --- VALIDATE DATA ---
            rec.status = RunStatus.VALIDATING
            validated_plan = self._validate_plan(plan)
            before = len(self._accumulated_examples)
            validated_plan = self._stack_training_examples(validated_plan)
            total_count = len(validated_plan.data_spec.examples)
            added = total_count - before
            exact_count = rec.config.training_sample_count
            yield await emit(
                EventKind.VALIDATION_COMPLETE,
                f"Plan validated: {total_count} examples "
                f"(+{added} new this iteration, stacked across {iteration} iter)",
                example_count=total_count,
                new_example_count=added,
                requested_training_sample_count=exact_count,
                hyperparams=validated_plan.hyperparams.model_dump(),
            )

            # --- TRAIN ---
            rec.status = RunStatus.TRAINING
            if warning := await self._acquire("gpu"):
                yield await emit(EventKind.LOG, warning)
            yield await emit(EventKind.TRAINING_STARTED, "Training started", backend=self._backend.name)

            artifact: ArtifactMeta | None = None
            train_cost = 0.0
            train_duration = 0.0
            train_error: str | None = None
            remote_payload = None
            remote_eval_records = None

            try:
                if rec.config.training_backend in {"remote", "unsloth", "local"}:
                    remote_payload = build_training_config_payload(
                        run_id=rid,
                        iteration=iteration,
                        run_config=rec.config,
                        plan=validated_plan,
                        eval_set=self._eval_set,
                    )

                stream = await self._backend.train(
                    rid,
                    iteration,
                    validated_plan,
                    rec.config.base_model_id,
                    remote_payload=remote_payload,
                )
                async with aclosing(stream):
                    async for progress in stream:
                        if self._cancelled:
                            break
                        if progress.get("done"):
                            artifact = progress.get("artifact")
                            train_cost = progress.get("cost_usd", 0.0)
                            train_duration = progress.get("duration_s", 0.0)
                            remote_eval_records = progress.get("eval_records")
                        else:
                            if progress.get("resources") is not None:
                                yield await emit(EventKind.RESOURCE_STATUS,
                                                 progress.get("message", "Resource availability updated"),
                                                 resources=progress["resources"], settings=progress.get("settings", {}))
                                continue
                            message = str(progress.get("message", "")).strip()
                            if message and "step" not in progress and "loss" not in progress:
                                phase = str(progress.get("phase", "remote"))
                                yield await emit(EventKind.LOG, f"[{self._backend.name}:{phase}] {message}")
                            else:
                                yield await emit(EventKind.TRAINING_PROGRESS, "", **{k: v for k, v in progress.items() if k != "done"})
            except Exception as exc:
                train_error = str(exc)

            if self._cancelled:
                yield await self._finish(rec, RunStatus.CANCELLED, StopReason.CANCELLED, emit)
                return

            if train_error or artifact is None:
                yield await emit(
                    EventKind.LOG,
                    f"[train] iteration {iteration} failed: "
                    f"{train_error or 'No artifact produced'}",
                )
                it_result = IterationResult(
                    iteration=iteration,
                    plan=validated_plan,
                    agent_fallback_used=fallback,
                    error=train_error or "No artifact produced",
                    started_at=_now(),
                    finished_at=_now(),
                )
                rec.iterations.append(it_result)
                await self._store.save(rec)

                failed_count = sum(1 for it in rec.iterations if it.error is not None)
                if failed_count >= 3:  # repeated training failures
                    rec.error = it_result.error
                    yield await self._finish(rec, RunStatus.FAILED, StopReason.TRAINING_FAILURE, emit)
                    return
                continue

            yield await emit(EventKind.TRAINING_COMPLETE, "Training complete", cost_usd=train_cost, duration_s=train_duration, is_mock=self._backend.is_mock)

            # Save mock artifact
            if self._backend.is_mock:
                artifact = await self._artifacts.save_mock(artifact)

            # --- EVAL ---
            rec.status = RunStatus.EVALUATING
            yield await emit(EventKind.EVAL_STARTED, "Running locked evaluation")

            # Integrity check before eval
            try:
                self._eval_set._assert_integrity()
            except RuntimeError:
                yield await self._finish(rec, RunStatus.FAILED, StopReason.EVAL_HASH_CHANGED, emit)
                return

            try:
                if remote_eval_records is not None:
                    score, eval_results, eval_responses = await self._judge_remote_evals(
                        remote_eval_records
                    )
                else:
                    score, eval_results, eval_responses = await self._eval(
                        adapter_path=artifact.adapter_path
                    )
            except Exception as exc:
                rec.error = f"Evaluation failed: {exc}"
                yield await emit(EventKind.LOG, rec.error)
                yield await self._finish(rec, RunStatus.FAILED, StopReason.BACKEND_UNAVAILABLE, emit)
                return
            artifact = artifact.model_copy(update={"score": score})
            if snapshot := getattr(self._runner, "last_resources", None):
                yield await emit(EventKind.RESOURCE_STATUS, "Evaluation resource availability",
                                 resources=snapshot["resources"], settings=snapshot.get("settings", {}))
            yield await emit(
                EventKind.EVAL_COMPLETE, f"Eval score: {score:.3f}", score=score,
                sample_results=[r.model_dump() for r in eval_results],
            )
            errors = [r for r in eval_results if r.details.get("error")]
            if errors:
                yield await emit(EventKind.LOG,
                                 f"{len(errors)}/{len(eval_results)} outputs could not be scored: "
                                 f"{errors[0].details['error']}")
            if any(r.details.get("scoring_source") == "objective fallback" for r in eval_results):
                yield await emit(EventKind.LOG, "LLM evaluator unavailable; used locked objective scoring")

            # --- ADAPTIVE ---
            adaptive_score = None
            if self._adaptive_set:
                rec.status = RunStatus.ADAPTIVE
                a_score, _, _ = await self._eval(adapter_path=artifact.adapter_path, eval_set=self._adaptive_set)
                adaptive_score = a_score
                yield await emit(EventKind.ADAPTIVE_COMPLETE, f"Adaptive score: {a_score:.3f}", score=a_score)

            judge_report = None
            retrain_decision: RetrainDecision | None = None
            if self._judge is not None:
                if warning := await self._acquire("judge"):
                    yield await emit(EventKind.LOG, warning)
                source = "heuristic" if self._judge.is_mock else _label(self._judge, "DigitalOcean")
                yield await emit(EventKind.JUDGE_STARTED, f"{source} judge reviewing iteration {iteration}")
                judge_report = await self._judge.judge(
                    goal=rec.config.goal,
                    task_name=rec.config.task_name,
                    base_model_id=rec.config.base_model_id,
                    iteration=iteration,
                    score=score,
                    adaptive_score=adaptive_score,
                    plan=validated_plan,
                    eval_results=eval_results,
                    responses=eval_responses,
                )
                yield await emit(
                    EventKind.JUDGE_COMPLETE,
                    f"Judge rating: {judge_report.rating:.2f}",
                    rating=judge_report.rating,
                    summary=judge_report.summary,
                    strengths=judge_report.strengths,
                    weaknesses=judge_report.weaknesses,
                    recommended_focus=judge_report.recommended_focus,
                    source=judge_report.source,
                    is_mock=judge_report.is_mock,
                )

                if self._retrain_advisor is not None:
                    if warning := await self._acquire("planner"):
                        yield await emit(EventKind.LOG, warning)
                    retrain_decision = await self._retrain_advisor.decide(
                        goal=rec.config.goal or rec.config.task_name,
                        rating=judge_report.rating,
                        target_score=rec.config.target_score,
                        iteration=iteration,
                        max_iterations=rec.config.max_iterations,
                        judge_report=judge_report,
                    )
                    advisor_name = self._advisor_name(retrain_decision)
                    yield await emit(
                        EventKind.RETRAIN_DECISION_RECEIVED,
                        (
                            f"{advisor_name} recommends retrain"
                            if retrain_decision.retrain_recommended
                            else f"{advisor_name} accepts current adapter"
                        ),
                        retrain_recommended=retrain_decision.retrain_recommended,
                        confidence=retrain_decision.confidence,
                        reason=retrain_decision.reason,
                        suggested_focus=retrain_decision.suggested_focus,
                        source=retrain_decision.source,
                        is_mock=retrain_decision.is_mock,
                    )

            # --- RECORD ITERATION ---
            it_result = IterationResult(
                iteration=iteration,
                plan=validated_plan,
                agent_fallback_used=fallback,
                training_cost_usd=train_cost,
                training_duration_s=train_duration,
                eval_results=eval_results,
                score=score,
                adaptive_score=adaptive_score,
                judge_report=judge_report,
                retrain_decision=retrain_decision,
                artifact=artifact,
                started_at=_now(),
                finished_at=_now(),
            )
            rec.iterations.append(it_result)
            rec.total_cost_usd += train_cost

            # Update best
            if score > rec.best_score:
                rec.best_score = score
                rec.best_iteration = iteration
                yield await emit(EventKind.BEST_UPDATED, f"New best: {score:.3f} (iteration {iteration})", score=score, iteration=iteration)

            yield await emit(EventKind.ITERATION_COMPLETE, f"Iteration {iteration} done: {score:.3f}", score=score, best=rec.best_score)
            await self._store.save(rec)

            accepted = retrain_decision is not None and not retrain_decision.retrain_recommended
            if (
                accepted
                and not rec.config.require_retrain_approval  # a person decides otherwise
                and score <= baseline_score
                and score < rec.config.target_score
                and (judge_report is None or judge_report.rating < LOW_JUDGE_RATING)
            ):
                # Small local advisors sometimes accept an adapter that learned nothing. In an
                # autonomous run nobody checks, so that is never a reason to stop.
                yield await emit(
                    EventKind.LOG,
                    f"[guard] {self._advisor_name(retrain_decision)} accepted an adapter scoring "
                    f"{score:.3f}, no better than the baseline {baseline_score:.3f}; continuing",
                )
                accepted = False
            if accepted:
                # The judge/advisor thinks it's good enough — but don't auto-stop. Let the
                # user decide whether to keep training to make it even smarter.
                if rec.config.require_retrain_approval and iteration < rec.config.max_iterations:
                    rating = judge_report.rating if judge_report else score
                    summary = judge_report.summary if judge_report else ""
                    async for ev in self._keep_training_gate(emit, rating, summary, "judge accepted"):
                        yield ev
                    if self._cancelled:
                        yield await self._finish(rec, RunStatus.CANCELLED, StopReason.CANCELLED, emit)
                        return
                    if self._keep_training:
                        continue  # user wants another round
                advisor_name = self._advisor_name(retrain_decision)
                yield await emit(
                    EventKind.STOP_CONDITION_MET,
                    f"Stop: {advisor_name} accepted the current adapter (user did not request more)",
                    reason=StopReason.JUDGE_ACCEPTED.value,
                )
                yield await self._finish(rec, RunStatus.COMPLETE, StopReason.JUDGE_ACCEPTED, emit)
                return

            # When the judge still wants more training (below target), auto-iterate without
            # asking — the user is only prompted once the model is 'good enough' (judge
            # accepted / target reached), via the keep-training gate.

            # --- STOP CONDITIONS ---
            stop = self._check_stop(rec, score)
            if (
                stop == StopReason.TARGET_SCORE
                and rec.config.require_retrain_approval
                and iteration < rec.config.max_iterations
            ):
                # Hit the target — but the judge/target shouldn't stop you. Ask the user.
                async for ev in self._keep_training_gate(emit, score, "", "target score reached"):
                    yield ev
                if self._cancelled:
                    yield await self._finish(rec, RunStatus.CANCELLED, StopReason.CANCELLED, emit)
                    return
                if self._keep_training:
                    continue  # user wants to keep going
            if stop:
                yield await emit(EventKind.STOP_CONDITION_MET, f"Stop: {stop.value}", reason=stop.value)
                yield await self._finish(rec, RunStatus.COMPLETE, stop, emit)
                return

        if rec.iterations and all(it.error is not None for it in rec.iterations):
            # Every iteration failed to train: there is no adapter, so this is not a success.
            rec.error = rec.iterations[-1].error
            yield await self._finish(rec, RunStatus.FAILED, StopReason.TRAINING_FAILURE, emit)
            return
        yield await self._finish(rec, RunStatus.COMPLETE, StopReason.MAX_ITERATIONS, emit)

    def _advisor_name(self, decision: RetrainDecision) -> str:
        if decision.is_mock:
            return "heuristic advisor"
        return _label(self._retrain_advisor, "MiniMax")

    async def _eval(
        self, adapter_path: str | None, eval_set: LockedEvalSet | None = None
    ) -> tuple[float, list[EvalResult], dict[str, str]]:
        es = eval_set or self._eval_set
        prompts = es.prompts_only()
        responses = await self._runner.run_batch(prompts, adapter_path=adapter_path)
        missing = [s.sample_id for s in es.samples if s.sample_id not in responses]
        if missing:
            raise RuntimeError(f"Model runner omitted evaluation outputs: {', '.join(missing)}")
        score, results = self._evaluator(es.samples, responses)
        for sample, result in zip(es.samples, results, strict=True):
            result.details.update(input=sample.prompt[:800], actual=responses[sample.sample_id][:800])
        return score, results, responses

    def _get_llm_judge(self):
        if self._llm_judge is None:
            from evolora.config import get_config
            from evolora.evaluation.llm_judge import get_llm_judge_evaluator

            cfg = get_config()
            self._llm_judge = get_llm_judge_evaluator(
                api_key=cfg.digital_ocean_model_access_key,
                base_url=cfg.digitalocean_inference_base_url,
                model=cfg.digital_ocean_judge_model,
            )
        return self._llm_judge

    def _eval_records_for_approval(self, samples: list[EvalSample]) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for sample in samples:
            expected = sample.expected
            records.append(
                {
                    "input": sample.prompt,
                    "expected": (
                        json.dumps(expected, sort_keys=True)
                        if isinstance(expected, (dict, list))
                        else str(expected)
                    ),
                }
            )
        return records

    async def _judge_remote_evals(
        self, records: list[dict]
    ) -> tuple[float, list[EvalResult], dict[str, str]]:
        """Score VM-produced eval records [{input, expected, actual}] with the LLM-judge."""
        aggregate, judged = await self._get_llm_judge().judge(records)
        if not records:
            raise RuntimeError("Remote evaluation returned no records")
        # A judge outage is not evidence of a bad adapter. Score the locked JSON
        # answers locally instead, with the same evaluator used by local runs.
        if any(not isinstance(r.get("score"), (int, float)) for r in judged):
            samples = self._eval_set.samples
            if len(records) != len(samples) or any(
                r.get("input") != s.prompt for r, s in zip(records, samples, strict=True)
            ):
                raise RuntimeError("Remote evaluation records do not match the locked prompts")
            responses = {s.sample_id: str(r.get("actual") or "")
                         for s, r in zip(samples, records, strict=True)}
            score, results = self._evaluator(samples, responses)
            for result, record in zip(results, judged, strict=True):
                result.details["scoring_source"] = "objective fallback"
                result.details["judge_error"] = record.get("reason") or "judge unavailable"
            return score, results, responses
        results: list[EvalResult] = []
        responses: dict[str, str] = {}
        for index, record in enumerate(judged):
            sample_id = f"vm-{index + 1:03d}"
            raw_score = record.get("score")
            score10 = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
            results.append(
                EvalResult(
                    sample_id=sample_id,
                    score=max(0.0, min(1.0, score10 / 10.0)),
                    passed=score10 >= 7,
                    details={
                        "input": str(record.get("input", ""))[:200],
                        "actual": str(record.get("actual", ""))[:200],
                        "reason": str(record.get("reason", "")),
                    },
                )
            )
            responses[sample_id] = str(record.get("actual", ""))
        return aggregate, results, responses

    async def _plan(
        self,
        iteration,
        baseline_score,
        current_score,
        failures,
        training_sample_count: int | None,
        goal: str = "",
    ):
        if isinstance(self._planner, MiniMaxPlanner):
            return await self._planner.plan(
                iteration,
                baseline_score,
                current_score,
                failures,
                training_sample_count,
                goal,
            )
        return (
            self._planner.plan(
                iteration,
                baseline_score,
                current_score,
                failures,
                training_sample_count,
                goal,
            ),
            False,
        )

    def _validate_plan(self, plan: AgentPlan) -> AgentPlan:
        requested = self._config.training_sample_count
        if requested is None:
            return plan

        examples = list(plan.data_spec.examples[:requested])
        # Pad up to the requested count by CYCLING the agent's real, on-goal examples rather than
        # injecting off-domain customer-spending filler — otherwise a JSON/SQL/etc. run gets
        # polluted with the wrong task. Fall back to the synthetic example only if the agent
        # produced nothing at all.
        on_goal = list(plan.data_spec.examples)
        i = 0
        while len(examples) < requested:
            if on_goal:
                examples.append(on_goal[i % len(on_goal)])
                i += 1
            else:
                examples.append(self._synthetic_training_example(len(examples) + 1))

        data_spec = plan.data_spec.model_copy(update={"examples": examples, "max_examples": requested})
        return plan.model_copy(update={"data_spec": data_spec})

    def _stack_training_examples(self, plan: AgentPlan) -> AgentPlan:
        """Accumulate training examples across iterations so the set keeps growing:
        iteration N trains on this round's examples PLUS every prior round's, so the
        count stacks by ~training_sample_count each iteration (30 -> 60 -> 90 …)."""
        self._accumulated_examples.extend(plan.data_spec.examples)
        data_spec = plan.data_spec.model_copy(
            update={
                "examples": list(self._accumulated_examples),
                "max_examples": len(self._accumulated_examples),
            }
        )
        return plan.model_copy(update={"data_spec": data_spec})

    def _synthetic_training_example(self, index: int) -> dict[str, str]:
        alice_a = 100 + index
        alice_b = 200 + index
        bob = 50 + index
        alice_total = alice_a + alice_b
        revenue = alice_total + bob
        return {
            "prompt": (
                f'Customers: [{{"name":"Alice","purchases":[{alice_a},{alice_b}]}}, '
                f'{{"name":"Bob","purchases":[{bob}]}}]. Summarize.'
            ),
            "completion": (
                f'{{"top_customer":"Alice","top_customer_total":{alice_total},'
                f'"customer_count":2,"total_revenue":{revenue},'
                f'"summary":"Alice leads with ${alice_total} in purchases."}}'
            ),
        }

    def _get_failures(self, rec: RunRecord) -> list[EvalResult]:
        if not rec.iterations:
            return [r for r in self._baseline_results if not r.passed]
        last = rec.iterations[-1]
        return [r for r in last.eval_results if not r.passed]

    async def _run_baseline(self, rec: RunRecord, rid: str, emit):
        """Run the baseline eval (remote VM or local). Stores the score in
        self._last_baseline_score and success in self._baseline_ok; yields progress events."""
        self._baseline_ok = True
        if warning := await self._acquire("gpu"):
            yield await emit(EventKind.LOG, warning)
        if rec.config.training_backend == "remote" and callable(
            getattr(self._backend, "evaluate_base", None)
        ):
            try:
                baseline_payload = build_baseline_config_payload(
                    run_id=rid,
                    run_config=rec.config,
                    eval_set=self._eval_set,
                )
                remote_baseline_records = None
                baseline_stream = await self._backend.evaluate_base(baseline_payload)  # type: ignore[attr-defined]
                async for progress in baseline_stream:
                    if progress.get("done"):
                        remote_baseline_records = progress.get("eval_records")
                        continue
                    message = str(progress.get("message", "")).strip()
                    if message:
                        phase = str(progress.get("phase", "baseline"))
                        yield await emit(EventKind.LOG, f"[{self._backend.name}:{phase}] {message}")
                if remote_baseline_records is None:
                    raise RuntimeError("Remote baseline did not return eval records")
                self._last_baseline_score, self._baseline_results, _ = await self._judge_remote_evals(
                    remote_baseline_records
                )
            except Exception as exc:
                rec.error = str(exc)
                yield await emit(EventKind.LOG, f"Remote baseline failed: {exc}")
                self._baseline_ok = False
        else:
            try:
                self._last_baseline_score, self._baseline_results, _ = await self._eval(adapter_path=None)
            except Exception as exc:
                rec.error = str(exc)
                yield await emit(EventKind.LOG, f"Baseline evaluation failed: {exc}")
                self._baseline_ok = False

    async def _keep_training_gate(self, emit, rating: float, summary: str, reason: str):
        """Ask the user whether to keep training to make the model smarter instead of
        auto-stopping when the judge/target says 'good enough'. Stores the answer in
        self._keep_training (True = keep going, False = stop)."""
        self._approval_future = asyncio.get_running_loop().create_future()
        yield await emit(
            EventKind.USER_APPROVAL_REQUIRED,
            f"Good enough ({reason}, rating {rating:.2f}) — keep training to make it smarter? "
            "YES = another round, NO = stop here.",
            rating=rating,
            summary=summary,
            approval_type="keep_training",
        )
        keep = await self._approval_future
        self._approval_future = None
        self._keep_training = keep
        yield await emit(
            EventKind.USER_APPROVAL_RECEIVED,
            "User chose to keep training" if keep else "User accepted the model — stopping",
            approved=keep,
            approval_type="keep_training",
        )

    def _check_stop(self, rec: RunRecord, score: float) -> StopReason | None:
        cfg = rec.config
        if score >= cfg.target_score:
            return StopReason.TARGET_SCORE
        if rec.no_improvement_count() >= cfg.patience:
            return StopReason.PATIENCE
        return None

    async def _finish(self, rec: RunRecord, status: RunStatus, reason: StopReason, emit) -> Event:
        rec.status = status
        rec.stop_reason = reason
        rec.finished_at = _now()
        await self._store.save(rec)
        kind = {
            RunStatus.COMPLETE: EventKind.RUN_COMPLETE,
            RunStatus.FAILED: EventKind.RUN_FAILED,
            RunStatus.CANCELLED: EventKind.RUN_CANCELLED,
        }.get(status, EventKind.RUN_COMPLETE)
        return Event(
            kind=kind,
            run_id=rec.run_id,
            message=f"Run {status.value}: {reason.value}",
            data={"reason": reason.value, "best_score": rec.best_score, "best_iteration": rec.best_iteration},
        )
