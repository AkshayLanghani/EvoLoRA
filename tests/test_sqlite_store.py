"""Tests for the SQLite run store."""

from datetime import UTC, datetime, timedelta

from evolora.models.core import RunConfig, RunRecord, RunStatus, StopReason
from evolora.persistence.sqlite_store import SqliteRunStore


def _record(goal: str, started: datetime) -> RunRecord:
    return RunRecord(config=RunConfig(goal=goal), started_at=started)


async def test_save_and_get_roundtrip(tmp_path):
    store = SqliteRunStore(tmp_path / "runs.db")
    rec = _record("strict json", datetime.now(UTC))
    rec.baseline_score = 0.4
    await store.save(rec)

    loaded = await store.get(rec.run_id)
    assert loaded is not None
    assert loaded.run_id == rec.run_id
    assert loaded.config.goal == "strict json"
    assert loaded.baseline_score == 0.4


async def test_save_updates_existing_run(tmp_path):
    store = SqliteRunStore(tmp_path / "runs.db")
    rec = _record("sql", datetime.now(UTC))
    await store.save(rec)
    rec.status = RunStatus.COMPLETE
    rec.stop_reason = StopReason.TARGET_SCORE
    await store.save(rec)

    runs = await store.list_runs()
    assert len(runs) == 1
    assert runs[0].status == RunStatus.COMPLETE
    assert runs[0].stop_reason == StopReason.TARGET_SCORE


async def test_list_is_newest_first_and_limited(tmp_path):
    store = SqliteRunStore(tmp_path / "runs.db")
    now = datetime.now(UTC)
    for i in range(3):
        await store.save(_record(f"goal {i}", now + timedelta(minutes=i)))

    runs = await store.list_runs(limit=2)
    assert [r.config.goal for r in runs] == ["goal 2", "goal 1"]


async def test_history_survives_reopening(tmp_path):
    path = tmp_path / "nested" / "runs.db"
    rec = _record("persisted", datetime.now(UTC))
    await SqliteRunStore(path).save(rec)

    assert (await SqliteRunStore(path).get(rec.run_id)) is not None


async def test_unfinished_runs_are_marked_interrupted(tmp_path):
    store = SqliteRunStore(tmp_path / "runs.db")
    running = _record("cut off", datetime.now(UTC))
    running.status = RunStatus.EVALUATING
    done = _record("finished", datetime.now(UTC))
    done.status = RunStatus.COMPLETE
    await store.save(running)
    await store.save(done)

    assert await store.mark_interrupted() == 1
    fixed = await store.get(running.run_id)
    assert fixed.status == RunStatus.CANCELLED and "Interrupted" in fixed.error
    assert (await store.get(done.run_id)).status == RunStatus.COMPLETE
    assert await store.mark_interrupted() == 0


async def test_missing_run_returns_none(tmp_path):
    assert await SqliteRunStore(tmp_path / "runs.db").get("nope") is None
