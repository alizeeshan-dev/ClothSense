from __future__ import annotations

import argparse

from clothsense.config import DEFAULT_CONFIG_PATH, load_config
from clothsense.experiments import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ClothSense Phase 5 experiments")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--clean-only", action="store_true")
    mode.add_argument("--shifts-only", action="store_true")
    mode.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke:
        database = config.paths.results / "smoke_experiments.sqlite"
        selected_mode = "all"
        seeds = args.seed or [config.seeds.experiments[0]]
        sample_limit = args.samples or 128
    else:
        database = config.paths.database
        selected_mode = "clean" if args.clean_only else "shifts" if args.shifts_only else "all"
        seeds = args.seed
        sample_limit = args.samples

    report = run_experiments(
        config,
        mode=selected_mode,
        seeds=seeds,
        database_path=database,
        sample_limit=sample_limit,
        download=not args.no_download,
        skip_completed=not args.rerun_completed,
    )
    print(f"database={database}")
    print(f"inference_passes={report.inference_passes}")
    print(f"executed_runs={report.executed_runs}")
    print(f"skipped_runs={report.skipped_runs}")
    print(f"failed_runs={report.failed_runs}")


if __name__ == "__main__":
    main()
