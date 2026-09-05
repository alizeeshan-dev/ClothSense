from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_database(path: str | Path) -> Path:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                seed INTEGER NOT NULL,
                config_hash TEXT NOT NULL,
                checkpoint_path TEXT NOT NULL,
                best_epoch INTEGER NOT NULL,
                validation_loss REAL NOT NULL,
                validation_accuracy REAL NOT NULL,
                temperature REAL,
                created_at TEXT NOT NULL,
                UNIQUE(seed, config_hash)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scenarios (
                scenario_id INTEGER PRIMARY KEY AUTOINCREMENT,
                shift_name TEXT NOT NULL,
                severity_name TEXT NOT NULL,
                parameters TEXT NOT NULL,
                seed INTEGER NOT NULL,
                sample_count INTEGER NOT NULL,
                UNIQUE (shift_name, severity_name, parameters, seed)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                scenario_id INTEGER NOT NULL,
                method_name TEXT NOT NULL,
                alpha REAL,
                abstention_policy TEXT,
                abstention_threshold REAL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                error_message TEXT,
                FOREIGN KEY (model_id) REFERENCES models(model_id),
                FOREIGN KEY (scenario_id) REFERENCES scenarios(scenario_id)
            )
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_identity
            ON runs (
                model_id, scenario_id, method_name,
                COALESCE(alpha, -1.0), COALESCE(abstention_policy, ''),
                COALESCE(abstention_threshold, -1.0)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                class_name TEXT,
                metric_value REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id)"
        )
        connection.commit()
    return database_path


def record_model_metadata(path: str | Path, metadata: dict[str, Any]) -> int:
    database_path = initialize_database(path)
    created_at = _utc_now()
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO models (
                seed, config_hash, checkpoint_path, best_epoch,
                validation_loss, validation_accuracy, temperature, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
            ON CONFLICT(seed, config_hash) DO UPDATE SET
                checkpoint_path = excluded.checkpoint_path,
                best_epoch = excluded.best_epoch,
                validation_loss = excluded.validation_loss,
                validation_accuracy = excluded.validation_accuracy,
                temperature = NULL,
                created_at = excluded.created_at
            """,
            (
                int(metadata["seed"]),
                str(metadata["config_hash"]),
                str(metadata["checkpoint_path"]),
                int(metadata["best_epoch"]),
                float(metadata["validation_loss"]),
                float(metadata["validation_accuracy"]),
                created_at,
            ),
        )
        row = connection.execute(
            "SELECT model_id FROM models WHERE seed = ? AND config_hash = ?",
            (int(metadata["seed"]), str(metadata["config_hash"])),
        ).fetchone()
    if row is None:
        raise RuntimeError("Failed to store model metadata")
    return int(row[0])


def update_model_temperature(
    path: str | Path,
    *,
    seed: int,
    config_hash: str,
    temperature: float,
) -> int:
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive")
    database_path = initialize_database(path)
    with _connect(database_path) as connection:
        cursor = connection.execute(
            "UPDATE models SET temperature = ? WHERE seed = ? AND config_hash = ?",
            (float(temperature), int(seed), str(config_hash)),
        )
        if cursor.rowcount != 1:
            raise ValueError("Expected exactly one compatible model metadata row")
        row = connection.execute(
            "SELECT model_id FROM models WHERE seed = ? AND config_hash = ?",
            (int(seed), str(config_hash)),
        ).fetchone()
    if row is None:
        raise RuntimeError("Failed to update model temperature")
    return int(row[0])


def canonical_parameters(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def upsert_scenario(
    path: str | Path,
    *,
    shift_name: str,
    severity_name: str,
    parameters: dict[str, Any],
    seed: int,
    sample_count: int,
) -> int:
    database_path = initialize_database(path)
    encoded = canonical_parameters(parameters)
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO scenarios (
                shift_name, severity_name, parameters, seed, sample_count
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(shift_name, severity_name, parameters, seed)
            DO UPDATE SET sample_count = excluded.sample_count
            """,
            (shift_name, severity_name, encoded, int(seed), int(sample_count)),
        )
        row = connection.execute(
            """
            SELECT scenario_id FROM scenarios
            WHERE shift_name = ? AND severity_name = ?
              AND parameters = ? AND seed = ?
            """,
            (shift_name, severity_name, encoded, int(seed)),
        ).fetchone()
    if row is None:
        raise RuntimeError("Failed to store scenario metadata")
    return int(row[0])


def get_model_id(path: str | Path, seed: int) -> int:
    database_path = initialize_database(path)
    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT model_id FROM models WHERE seed = ?", (int(seed),)
        ).fetchone()
    if row is None:
        raise KeyError(f"No model metadata exists for seed {seed}")
    return int(row[0])


def start_run(
    path: str | Path,
    *,
    model_id: int,
    scenario_id: int,
    method_name: str,
    alpha: float | None = None,
    abstention_policy: str | None = None,
    abstention_threshold: float | None = None,
    skip_completed: bool = True,
) -> tuple[int, bool]:
    database_path = initialize_database(path)
    with _connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT run_id, status FROM runs
            WHERE model_id = ? AND scenario_id = ? AND method_name = ?
              AND alpha IS ? AND abstention_policy IS ?
              AND abstention_threshold IS ?
            """,
            (
                int(model_id), int(scenario_id), method_name, alpha,
                abstention_policy, abstention_threshold,
            ),
        ).fetchone()
        if row is not None:
            run_id, status = int(row[0]), str(row[1])
            if skip_completed and status == "completed":
                return run_id, False
            connection.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            connection.execute(
                """
                UPDATE runs SET started_at = ?, completed_at = NULL,
                    status = 'started', error_message = NULL
                WHERE run_id = ?
                """,
                (_utc_now(), run_id),
            )
        else:
            cursor = connection.execute(
                """
                INSERT INTO runs (
                    model_id, scenario_id, method_name, alpha,
                    abstention_policy, abstention_threshold,
                    started_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'started')
                """,
                (
                    int(model_id), int(scenario_id), method_name, alpha,
                    abstention_policy, abstention_threshold, _utc_now(),
                ),
            )
            run_id = int(cursor.lastrowid)
    return run_id, True


def complete_run(
    path: str | Path,
    run_id: int,
    metrics: Iterable[tuple[str, str | None, float | int | None]],
) -> None:
    rows = [
        (int(run_id), name, class_name, float(value))
        for name, class_name, value in metrics
        if value is not None
    ]
    with _connect(path) as connection:
        connection.execute("DELETE FROM metrics WHERE run_id = ?", (int(run_id),))
        if rows:
            connection.executemany(
                """
                INSERT INTO metrics (run_id, metric_name, class_name, metric_value)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
        connection.execute(
            """
            UPDATE runs SET status = 'completed', completed_at = ?, error_message = NULL
            WHERE run_id = ?
            """,
            (_utc_now(), int(run_id)),
        )


def fail_run(path: str | Path, run_id: int, error: str) -> None:
    with _connect(path) as connection:
        connection.execute(
            """
            UPDATE runs SET status = 'failed', completed_at = ?, error_message = ?
            WHERE run_id = ?
            """,
            (_utc_now(), str(error)[:1000], int(run_id)),
        )


def foreign_key_violations(path: str | Path) -> list[tuple[Any, ...]]:
    with _connect(path) as connection:
        return list(connection.execute("PRAGMA foreign_key_check").fetchall())
