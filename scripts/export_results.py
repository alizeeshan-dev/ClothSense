from __future__ import annotations

import argparse

from clothsense.config import DEFAULT_CONFIG_PATH, load_config
from clothsense.reporting import export_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild ClothSense CSVs from SQLite")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--database")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = export_results(
        args.database or config.paths.database,
        config.paths.results,
        expected_seeds=config.seeds.experiments,
    )
    for path in paths.__dict__.values():
        print(path)


if __name__ == "__main__":
    main()
