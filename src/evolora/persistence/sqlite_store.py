"""SQLite run store: durable local history with no database server to install."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from evolora.models.core import RunRecord, RunStatus, StopReason

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    status     TEXT NOT NULL,
    goal       TEXT NOT NULL,
    record     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_started_at ON runs (started_at DESC);
"""


class SqliteRunStore:
    """Stores each RunRecord as JSON, with a few columns for listing and filtering."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        # A short-lived connection per call keeps this safe across threads.
        return sqlite3.connect(self._path, timeout=10)

    async def save(self, record: RunRecord) -> None:
        await asyncio.to_thread(self._save, record)

    async def get(self, run_id: str) -> RunRecord | None:
        return await asyncio.to_thread(self._get, run_id)

    async def list_runs(self, limit: int = 50) -> list[RunRecord]:
        return await asyncio.to_thread(self._list, limit)

    def _save(self, record: RunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, started_at, status, goal, record)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status, goal = excluded.goal, record = excluded.record
                """,
                (
                    record.run_id,
                    record.started_at.isoformat(),
                    record.status.value,
                    record.config.goal,
                    record.model_dump_json(),
                ),
            )

    def _get(self, run_id: str) -> RunRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT record FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return RunRecord.model_validate_json(row[0]) if row else None

    async def mark_interrupted(self) -> int:
        """Close out runs left unfinished when the process stopped; returns how many."""
        return await asyncio.to_thread(self._mark_interrupted)

    def _mark_interrupted(self) -> int:
        terminal = {s.value for s in (RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.CANCELLED)}
        with self._connect() as conn:
            rows = conn.execute("SELECT record FROM runs").fetchall()
        changed = 0
        for (raw,) in rows:
            record = RunRecord.model_validate_json(raw)
            if record.status.value in terminal:
                continue
            record.status = RunStatus.CANCELLED
            record.stop_reason = StopReason.CANCELLED
            record.error = "Interrupted: EvoLoRA stopped before the run finished"
            record.finished_at = datetime.now(UTC)
            self._save(record)
            changed += 1
        return changed

    def _list(self, limit: int) -> list[RunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [RunRecord.model_validate_json(r[0]) for r in rows]
