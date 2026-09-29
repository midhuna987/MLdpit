"""Convert yearly NetCDF precipitation files into chunked Zarr stores.

 One Zarr store is written per calendar year, chunked along time at one day:
 24 steps for hourly data, 144 for 10-minute data.  A whole year is opened
 lazily and sliced year by year, so a file spanning a year boundary produces two
 stores.
 """

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from mldpit.config import parse_args_with_config

#: Default time chunk (one day) for each kind of input.
DEFAULT_CHUNKS = {"hourly": 24, "subhourly": 144}


def convert_file(
    nc_file: str | Path,
    output_dir: str | Path,
    output_template: str,
    time_chunk: int,
    experiment: str = "",
    overwrite: bool = True,
) -> list[Path]:
    """Split one NetCDF file into per-year Zarr stores."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    ds = xr.open_dataset(nc_file, chunks={"time": time_chunk})
    try:
        for year in np.unique(ds["time.year"].values):
            year = int(year)
            target = output_dir / output_template.format(
                year=year, experiment=experiment
            )
            if target.exists() and not overwrite:
                print(f"  exists, skipping: {target}")
                continue
            ds_year = ds.sel(time=ds["time.year"] == year).chunk({"time": time_chunk})
            ds_year.to_zarr(target, mode="w", consolidated=True)
            written.append(target)
            print(f"  wrote {target}")
            del ds_year
            gc.collect()
    finally:
        ds.close()
        del ds
        gc.collect()

    return written


def parse_years(spec: str) -> list[int]:
    """Parse ``"1960-2024"`` or ``"1960,1961,1975"`` into a list of years."""
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            first, last = part.split("-", 1)
            years.extend(range(int(first), int(last) + 1))
        elif part:
            years.append(int(part))
    return sorted(set(years))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mldpit-preprocess",
        description="Convert yearly NetCDF precipitation files into Zarr stores.",
    )
    parser.add_argument(
        "--kind", choices=sorted(DEFAULT_CHUNKS), required=True,
        help="hourly for the source fields, subhourly for the 10-minute targets.",
    )
    parser.add_argument("--input-dir", required=True, help="Directory of NetCDF input.")
    parser.add_argument("--output-dir", required=True, help="Directory for Zarr output.")
    parser.add_argument(
        "--years", required=True,
        help='Years to process, e.g. "1960-2024" or "2001,2002,2010".',
    )
    parser.add_argument(
        "--input-template",
        default="TOT_PREC_10min_{year}010100-{next_year}010100.ncz",
        help="Filename pattern for the input; fields: {year}, {next_year}, {experiment}.",
    )
    parser.add_argument(
        "--output-template",
        default="train_target_{year}_{experiment}_10mnts.zarr",
        help="Filename pattern for the output; fields: {year}, {experiment}.",
    )
    parser.add_argument("--experiment", default="", help="Experiment identifier.")
    parser.add_argument(
        "--time-chunk", type=int, default=None,
        help="Time chunk size; defaults to one day for the chosen --kind.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Leave Zarr stores that already exist untouched.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    time_chunk = args.time_chunk or DEFAULT_CHUNKS[args.kind]

    missing: list[str] = []
    for year in parse_years(args.years):
        filename = args.input_template.format(
            year=year, next_year=year + 1, experiment=args.experiment
        )
        path = Path(args.input_dir) / filename
        if not path.exists():
            missing.append(str(path))
            print(f"File not found: {path}", file=sys.stderr)
            continue
        print(f"Processing {path}")
        convert_file(
            path, args.output_dir, args.output_template, time_chunk,
            experiment=args.experiment, overwrite=not args.skip_existing,
        )

    if missing:
        print(f"\n{len(missing)} input file(s) missing.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
