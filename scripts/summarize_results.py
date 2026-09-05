from __future__ import annotations

import argparse

from clothsense.config import DEFAULT_CONFIG_PATH, load_config
from clothsense.research_summary import generate_research_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Phase 5 research summary from CSV files")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    config = load_config(args.config)
    policy = config.uncertainty.upload_demo_abstention_policy
    if policy is None:
        raise ValueError("Select upload_demo_abstention_policy in configuration first")
    path = generate_research_summary(
        config.paths.results,
        config.paths.results / "research_summary.md",
        demo_alpha=config.uncertainty.upload_demo_alpha,
        demo_policy=policy,
        demo_threshold=config.uncertainty.upload_demo_abstention_threshold,
    )
    print(path)


if __name__ == "__main__":
    main()
