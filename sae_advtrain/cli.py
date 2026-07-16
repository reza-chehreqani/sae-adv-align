"""Shared CLI plumbing for scripts/*.py."""
from __future__ import annotations

import argparse

from .config import ExperimentConfig, load_config


def parse_config_args(description: str) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="Dotted-key override, e.g. --set optim.epochs=50. Repeatable.",
    )
    args = parser.parse_args()
    return load_config(args.config, args.overrides)
