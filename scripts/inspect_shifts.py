from __future__ import annotations

import argparse

from clothsense.config import DEFAULT_CONFIG_PATH, load_config
from clothsense.data import load_raw_clean_test
from clothsense.plotting import plot_shift_verification_grid
from clothsense.shifts import build_shift_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Phase 4 shift inspection grid")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    clean_test = load_raw_clean_test(config, download=not args.no_download)
    scenarios = build_shift_scenarios(clean_test, config, seed=args.seed)
    output = plot_shift_verification_grid(
        scenarios,
        config.paths.plots / "shift_verification.png",
    )
    for generated in scenarios:
        scenario = generated.scenario
        print(
            f"{scenario.shift_name}/{scenario.severity_name}: "
            f"n={scenario.sample_count} parameters={dict(scenario.parameters)}"
        )
    print(f"grid={output}")


if __name__ == "__main__":
    main()

