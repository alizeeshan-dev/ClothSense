from __future__ import annotations

import argparse

from clothsense.config import DEFAULT_CONFIG_PATH, load_config
from clothsense.research_plots import generate_research_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 5 figures from persisted CSV files")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    config = load_config(args.config)
    for path in generate_research_plots(config.paths.results, config.paths.plots):
        print(path)


if __name__ == "__main__":
    main()
