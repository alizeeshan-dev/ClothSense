from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .abstention import (
    confidence_threshold_decision,
    hybrid_decision,
    singleton_set_decision,
)
from .calibration import softmax_predictions, temperature_scaled_predictions
from .config import ProjectConfig
from .conformal import class_conditional_prediction_sets, prediction_sets
from .data import CLASS_NAMES, load_raw_clean_test
from .evaluation import RuntimeMetrics, collect_logits_with_runtime
from .metrics import (
    classification_metrics,
    conformal_metrics,
    probability_metrics,
    selective_prediction_metrics,
)
from .reproducibility import seed_everything, select_device
from .shifts import GeneratedScenario, build_shift_scenarios
from .storage import (
    complete_run,
    fail_run,
    get_model_id,
    initialize_database,
    record_model_metadata,
    start_run,
    update_model_temperature,
    upsert_scenario,
)
from .training import configuration_hash, load_checkpoint, seed_artifact_paths
from .uncertainty_artifacts import file_sha256, load_uncertainty_artifact


@dataclass(frozen=True)
class RunSpec:
    method_name: str
    alpha: float | None = None
    abstention_policy: str | None = None
    abstention_threshold: float | None = None


@dataclass(frozen=True)
class ExperimentReport:
    inference_passes: int
    executed_runs: int
    skipped_runs: int
    failed_runs: int


def build_run_specs(config: ProjectConfig) -> tuple[RunSpec, ...]:
    specs = [RunSpec("raw_softmax"), RunSpec("temperature_scaled")]
    for alpha in config.uncertainty.alpha_values:
        specs.append(RunSpec("standard_conformal", alpha=float(alpha)))
        specs.append(RunSpec("class_conditional_conformal", alpha=float(alpha)))
    for threshold in config.uncertainty.abstention_thresholds:
        specs.append(
            RunSpec(
                "temperature_scaled",
                abstention_policy="confidence_threshold",
                abstention_threshold=float(threshold),
            )
        )
    for alpha in config.uncertainty.alpha_values:
        specs.append(
            RunSpec(
                "standard_conformal",
                alpha=float(alpha),
                abstention_policy="singleton_conformal",
            )
        )
        for threshold in config.uncertainty.abstention_thresholds:
            specs.append(
                RunSpec(
                    "standard_conformal",
                    alpha=float(alpha),
                    abstention_policy="hybrid",
                    abstention_threshold=float(threshold),
                )
            )
    return tuple(specs)


def _ensure_model_record(
    database_path: Path,
    config: ProjectConfig,
    seed: int,
    checkpoint: dict[str, object],
    temperature: float,
) -> int:
    try:
        return get_model_id(database_path, seed)
    except KeyError:
        model_id = record_model_metadata(
            database_path,
            {
                "seed": seed,
                "config_hash": checkpoint["config_hash"],
                "checkpoint_path": seed_artifact_paths(config, seed).checkpoint,
                "best_epoch": checkpoint["best_epoch"],
                "validation_loss": checkpoint["validation_loss"],
                "validation_accuracy": checkpoint["validation_accuracy"],
            },
        )
        update_model_temperature(
            database_path,
            seed=seed,
            config_hash=str(checkpoint["config_hash"]),
            temperature=temperature,
        )
        return model_id


def _aggregate_rows(values: dict[str, float | int | None]) -> list[tuple[str, None, float | int | None]]:
    return [(name, None, value) for name, value in values.items()]


def _classification_rows(result: dict) -> list[tuple[str, str | None, float | int | None]]:
    rows: list[tuple[str, str | None, float | int | None]] = [
        ("accuracy", None, result["accuracy"]),
        ("macro_f1", None, result["macro_f1"]),
    ]
    for item in result["per_class"]:
        class_name = str(item["class_name"])
        rows.extend(
            (metric, class_name, item[metric])
            for metric in ("precision", "recall", "f1")
        )
    for true_id, matrix_row in enumerate(result["confusion_matrix"]):
        for predicted_id, value in enumerate(matrix_row):
            rows.append((f"confusion_true_{true_id}_pred_{predicted_id}", None, value))
    return rows


def _probability_rows(result: dict) -> list[tuple[str, str | None, float | int | None]]:
    rows: list[tuple[str, str | None, float | int | None]] = _aggregate_rows(
        result["aggregate"]
    )
    for item in result["reliability_bins"]:
        class_name = f"bin:{item['bin_index']}"
        for name in (
            "lower_bound",
            "upper_bound",
            "sample_count",
            "average_confidence",
            "empirical_accuracy",
        ):
            rows.append((name, class_name, item[name]))
    return rows


def _conformal_rows(result: dict) -> list[tuple[str, str | None, float | int | None]]:
    rows: list[tuple[str, str | None, float | int | None]] = _aggregate_rows(
        result["aggregate"]
    )
    for item in result["per_class"]:
        class_name = str(item["class_name"])
        rows.append(("coverage", class_name, item["coverage"]))
        rows.append(("average_set_size", class_name, item["average_set_size"]))
    return rows


def _training_history_rows(path: Path) -> list[tuple[str, None, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history", payload)
    rows: list[tuple[str, None, float]] = []
    fields = ("train_loss", "validation_loss", "train_accuracy", "validation_accuracy")
    for index, epoch in enumerate(history["epoch"]):
        for field in fields:
            rows.append((f"training_{field}_epoch_{int(epoch):03d}", None, history[field][index]))
    return rows


def _selective_rows(result: dict) -> list[tuple[str, None, float | int | None]]:
    return _aggregate_rows(result)


def _load_clean_outputs(
    path: Path,
    sample_limit: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RuntimeMetrics | None]:
    with np.load(path, allow_pickle=False) as saved:
        logits = saved["logits"].astype(np.float32)
        labels = saved["labels"].astype(np.int64)
        predictions = saved["predictions"].astype(np.int64)
        runtime = (
            RuntimeMetrics(float(saved["evaluation_seconds"]), len(labels))
            if "evaluation_seconds" in saved.files
            else None
        )
    if logits.shape != (len(labels), len(CLASS_NAMES)) or predictions.shape != labels.shape:
        raise ValueError("Saved clean-test outputs have incompatible shapes")
    stop = len(labels) if sample_limit is None else min(sample_limit, len(labels))
    if runtime is not None and stop != len(labels):
        runtime = None
    return logits[:stop], labels[:stop], predictions[:stop], runtime


def _make_loader(dataset, config: ProjectConfig, sample_limit: int | None) -> DataLoader:
    if sample_limit is not None:
        dataset = Subset(dataset, range(min(sample_limit, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory and torch.cuda.is_available(),
        persistent_workers=config.training.num_workers > 0,
    )


def _scenario_selection(
    raw_clean,
    config: ProjectConfig,
    seed: int,
    mode: str,
) -> list[GeneratedScenario | None]:
    shifted = build_shift_scenarios(raw_clean, config, seed=seed)
    if mode == "clean":
        return [None]
    if mode == "shifts":
        return list(shifted)
    if mode == "all":
        return [None, *shifted]
    raise ValueError("mode must be one of: clean, shifts, all")


def _scenario_metadata(
    generated: GeneratedScenario | None,
    seed: int,
    config: ProjectConfig,
    sample_limit: int | None,
) -> tuple[str, str, dict, int]:
    if generated is None:
        count = config.data.test_size
        shift_name, severity, parameters = "clean", "clean", {}
    else:
        metadata = generated.scenario
        count = metadata.sample_count
        shift_name, severity, parameters = (
            metadata.shift_name,
            metadata.severity_name,
            dict(metadata.parameters),
        )
    if sample_limit is not None:
        count = min(count, sample_limit)
        parameters = {**parameters, "sample_limit": int(sample_limit)}
    return shift_name, severity, parameters, count


def _metric_rows_for_spec(
    spec: RunSpec,
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    raw,
    calibrated,
    standard_sets: dict[float, list[tuple[int, ...]]],
    conditional_sets: dict[float, list[tuple[int, ...]]],
    config: ProjectConfig,
    runtime: RuntimeMetrics | None,
    history_path: Path | None,
) -> list[tuple[str, str | None, float | int | None]]:
    if spec.abstention_policy is not None:
        if spec.abstention_policy == "confidence_threshold":
            decisions = [
                confidence_threshold_decision(int(prediction), float(confidence), float(spec.abstention_threshold))
                for prediction, confidence in zip(calibrated.predicted_classes, calibrated.confidence)
            ]
        elif spec.abstention_policy == "singleton_conformal":
            decisions = [singleton_set_decision(value) for value in standard_sets[float(spec.alpha)]]
        elif spec.abstention_policy == "hybrid":
            decisions = [
                hybrid_decision(int(prediction), float(confidence), value, float(spec.abstention_threshold))
                for prediction, confidence, value in zip(
                    calibrated.predicted_classes,
                    calibrated.confidence,
                    standard_sets[float(spec.alpha)],
                )
            ]
        else:
            raise ValueError(f"Unknown abstention policy: {spec.abstention_policy}")
        return _selective_rows(selective_prediction_metrics(labels, decisions))

    if spec.method_name == "raw_softmax":
        result = classification_metrics(labels, predictions)
        rows = _classification_rows(result)
        rows.extend(
            _probability_rows(
                probability_metrics(raw.probabilities, labels, config.uncertainty.calibration_bin_edges)
            )
        )
        if runtime is not None:
            rows.extend(_aggregate_rows(runtime.as_dict()))
        if history_path is not None:
            rows.extend(_training_history_rows(history_path))
        return rows
    if spec.method_name == "temperature_scaled":
        return _probability_rows(
            probability_metrics(
                calibrated.probabilities,
                labels,
                config.uncertainty.calibration_bin_edges,
            )
        )
    if spec.method_name == "standard_conformal":
        result = conformal_metrics(
            standard_sets[float(spec.alpha)], labels, target_coverage=1.0 - float(spec.alpha)
        )
        return _conformal_rows(result)
    if spec.method_name == "class_conditional_conformal":
        result = conformal_metrics(
            conditional_sets[float(spec.alpha)], labels, target_coverage=1.0 - float(spec.alpha)
        )
        return _conformal_rows(result)
    raise ValueError(f"Unknown method: {spec.method_name}")


def run_experiments(
    config: ProjectConfig,
    *,
    mode: str = "all",
    seeds: Sequence[int] | None = None,
    database_path: str | Path | None = None,
    sample_limit: int | None = None,
    download: bool = True,
    skip_completed: bool = True,
) -> ExperimentReport:
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("sample_limit must be positive")
    database = initialize_database(database_path or config.paths.database)
    selected_seeds = tuple(config.seeds.experiments if seeds is None else seeds)
    unknown = set(selected_seeds) - set(config.seeds.experiments)
    if unknown:
        raise ValueError(f"Seeds are not configured experiment seeds: {sorted(unknown)}")
    specs = build_run_specs(config)
    device = select_device(config.device)
    raw_clean = load_raw_clean_test(config, download=download)
    run_hash = configuration_hash(config)
    inference_passes = executed = skipped = failed = 0

    for seed in selected_seeds:
        seed_everything(seed, config.device.deterministic_algorithms)
        artifacts = seed_artifact_paths(config, seed)
        model, checkpoint = load_checkpoint(artifacts.checkpoint, device)
        if int(checkpoint["seed"]) != seed or checkpoint["config_hash"] != run_hash:
            raise ValueError(f"Checkpoint for seed {seed} is incompatible")
        fitted, _ = load_uncertainty_artifact(
            artifacts.uncertainty,
            expected_seed=seed,
            expected_config_hash=run_hash,
            expected_checkpoint_sha256=file_sha256(artifacts.checkpoint),
        )
        model_id = _ensure_model_record(database, config, seed, checkpoint, fitted.temperature)

        for generated in _scenario_selection(raw_clean, config, seed, mode):
            shift_name, severity, parameters, sample_count = _scenario_metadata(
                generated, seed, config, sample_limit
            )
            scenario_id = upsert_scenario(
                database,
                shift_name=shift_name,
                severity_name=severity,
                parameters=parameters,
                seed=seed,
                sample_count=sample_count,
            )
            pending: list[tuple[RunSpec, int]] = []
            for spec in specs:
                run_id, should_execute = start_run(
                    database,
                    model_id=model_id,
                    scenario_id=scenario_id,
                    method_name=spec.method_name,
                    alpha=spec.alpha,
                    abstention_policy=spec.abstention_policy,
                    abstention_threshold=spec.abstention_threshold,
                    skip_completed=skip_completed,
                )
                if should_execute:
                    pending.append((spec, run_id))
                else:
                    skipped += 1
            if not pending:
                continue

            completed_in_batch: set[int] = set()
            try:
                if generated is None:
                    logits, labels, predictions, runtime = _load_clean_outputs(
                        artifacts.clean_outputs, sample_limit
                    )
                else:
                    loader = _make_loader(generated.dataset, config, sample_limit)
                    logits, labels, predictions, runtime = collect_logits_with_runtime(
                        model, loader, device
                    )
                    inference_passes += 1
                logits_tensor = torch.from_numpy(logits)
                raw = softmax_predictions(logits_tensor)
                calibrated = temperature_scaled_predictions(logits_tensor, fitted.temperature)
                standard_sets = {
                    float(alpha): prediction_sets(
                        calibrated.probabilities, fitted.standard_thresholds[float(alpha)]
                    )
                    for alpha in config.uncertainty.alpha_values
                }
                conditional_sets = {
                    float(alpha): class_conditional_prediction_sets(
                        calibrated.probabilities,
                        fitted.class_conditional_thresholds[float(alpha)],
                    )
                    for alpha in config.uncertainty.alpha_values
                }
                for spec, run_id in pending:
                    rows = _metric_rows_for_spec(
                        spec,
                        labels=labels,
                        predictions=predictions,
                        raw=raw,
                        calibrated=calibrated,
                        standard_sets=standard_sets,
                        conditional_sets=conditional_sets,
                        config=config,
                        runtime=runtime if spec.method_name == "raw_softmax" else None,
                        history_path=(
                            artifacts.history
                            if generated is None and spec.method_name == "raw_softmax"
                            else None
                        ),
                    )
                    complete_run(database, run_id, rows)
                    completed_in_batch.add(run_id)
                    executed += 1
            except Exception as error:
                for _, run_id in pending:
                    if run_id not in completed_in_batch:
                        fail_run(database, run_id, str(error))
                        failed += 1
                raise
    return ExperimentReport(inference_passes, executed, skipped, failed)
