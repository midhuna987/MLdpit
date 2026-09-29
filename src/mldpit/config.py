"""Configuration loading shared by the command-line entry points.

Every entry point accepts ``--config path.yaml``.  Keys in that file set the
defaults for the corresponding command-line flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping, returning an empty dict for an empty file."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse ``argv``, applying ``--config`` as a source of defaults."""
    parser.add_argument(
        "--config", type=Path, default=None,
        help="YAML configuration file. Command-line flags override its values.",
    )

    # A separate, minimal parser finds --config. The main parser cannot be used
    # here: parse_known_args still enforces required=True, so it would reject a
    # command line whose required values are supplied by the file.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    if pre_args.config is not None:
        config = load_yaml(pre_args.config)
        valid = {action.dest for action in parser._actions}
        unknown = sorted(set(config) - valid)
        if unknown:
            raise SystemExit(
                f"{pre_args.config}: unrecognised configuration key(s): "
                + ", ".join(unknown)
            )
        parser.set_defaults(**config)
        # set_defaults does not clear required=True, so an argument supplied by
        # the file would still be reported as missing.
        for action in parser._actions:
            if action.dest in config:
                action.required = False

    return parser.parse_args(argv)


def resolve_paths(
    directory: str | Path,
    template: str,
    years: Sequence[int],
    **fields: Any,
) -> list[str]:
    """Expand a filename template over a range of years.

    ``template`` is a :meth:`str.format` pattern such as
    ``"train_source_{year}_{experiment}_1hr_hoursum.zarr"``.
    """
    directory = Path(directory)
    return [str(directory / template.format(year=year, **fields)) for year in years]
