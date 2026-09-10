from __future__ import annotations

import argparse

from .runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the single-station PV P1-P16 component and fusion experiment"
    )
    parser.add_argument("--config", required=True, help="YAML configuration path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()

