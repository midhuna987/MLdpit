#inference : hourly precipitation to 10-minute precipitation.

from __future__ import annotations
from pytorch_lightning.plugins.environments import LightningEnvironment
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader

from mldpit.config import parse_args_with_config, resolve_paths
from mldpit.dataset import VAR_CANDIDATES, LazyXarraySourceOnlyDataset
from mldpit.model import LightningSRUNet, softmax_constraint

_NETCDF_ENCODING_KEYS = {
    "dtype", "_FillValue", "units", "calendar", "scale_factor", "add_offset",
    "zlib", "complevel", "shuffle", "contiguous",
}


def _clean_encoding(encoding: dict, fmt: str) -> dict:
    encoding = dict(encoding)
    if fmt == "netcdf":
        return {k: v for k, v in encoding.items() if k in _NETCDF_ENCODING_KEYS}
    for key in ("chunks", "preferred_chunks"):
        encoding.pop(key, None)
    return encoding


def _infer_format(path: str | Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    return "netcdf" if str(path).endswith((".nc", ".nc4", ".netcdf")) else "zarr"


def _canonical_source(path, source_var: str) -> xr.Dataset:
    ds = xr.open_zarr(path[0] if isinstance(path, (list, tuple)) else path)
    for candidate in VAR_CANDIDATES:
        if candidate in ds.data_vars:
            return ds.rename({candidate: source_var}) if candidate != source_var else ds
    raise KeyError(f"none of {VAR_CANDIDATES} in {list(ds.data_vars)}")


class _PredictModel(LightningSRUNet):
    

    LOG_EPS = 1e-6
    SUBTRACT_LOG_EPS = True

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        total = self.hourly_total(x)
        if self.SUBTRACT_LOG_EPS:
            total = torch.clamp(total - self.LOG_EPS, min=0.0)
        with torch.no_grad():
            return softmax_constraint(10.0 ** self.model(x), total, dim=1)


def build_output(
    values: np.ndarray,
    valid_times: np.ndarray,
    source_ds: xr.Dataset,
    source_var: str,
    n_sub_intervals: int,
    time_bounds: str,
) -> xr.Dataset:
    """Assemble the predicted fields into a CF-style dataset."""
    source_da = source_ds[source_var]
    spatial_dims = source_da.dims[-2:]

    coords: dict = {"time": valid_times}
    for dim in spatial_dims:
        if dim in source_ds.coords:
            coords[dim] = source_ds[dim]
    for name in ("lat", "lon"):
        if name in source_ds.coords and source_ds[name].dims == spatial_dims:
            coords[name] = xr.Variable(
                spatial_dims, source_ds[name].values, attrs=dict(source_ds[name].attrs)
            )

    prediction = xr.DataArray(
        values, dims=("time", *spatial_dims), coords=coords, name=source_var
    )
    prediction.attrs = dict(source_da.attrs)

    step = 60 // n_sub_intervals
    if time_bounds == "legacy":
        # Interval-end convention: [t - step, t]. 
        lower = valid_times - np.timedelta64(step, "m")
        upper = valid_times
    else:
        # Interval-centre convention: [t - step/2, t + step/2].
        half = np.timedelta64(step // 2, "m")
        lower, upper = valid_times - half, valid_times + half

    bounds = xr.DataArray(
        np.stack([lower, upper], axis=-1),
        dims=("time", "bnds"),
        coords={"time": valid_times, "bnds": [0, 1]},
        name="time_bnds",
    )
    if "time_bnds" in source_ds:
        bounds.attrs = dict(source_ds["time_bnds"].attrs)

    data_vars = {source_var: prediction, "time_bnds": bounds}

    # Carry the grid-mapping variable (rotated_pole) across, so that the
    # grid_mapping attribute copied onto the precipitation variable.
    grid_mapping = source_da.attrs.get("grid_mapping")
    if grid_mapping and grid_mapping in source_ds.variables:
        data_vars[grid_mapping] = source_ds[grid_mapping].reset_coords(drop=True)

    result = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(source_ds.attrs))

    # CF link between the time axis and its bounds. Without it, tools such as
    # CDO treat time_bnds as a second data variable instead of as the interval
    # each 10-minute value represents.
    result["time"].attrs["bounds"] = "time_bnds"
    result["time_bnds"].attrs.pop("units", None)
    result["time_bnds"].attrs.pop("calendar", None)
    
    for name in list(result.coords):
        target = result[name].attrs.get("bounds")
        if not target or target in result.variables:
            continue
        candidate = source_ds[target] if target in source_ds.variables else None
        if candidate is not None and all(
            d not in result.sizes or result.sizes[d] == candidate.sizes[d]
            for d in candidate.dims
        ):
            copied = candidate.reset_coords(drop=True)            
            copied.attrs = {}
            
            copied.encoding = {"_FillValue": None, "coordinates": None}
            result[target] = copied
        else:
            result[name].attrs.pop("bounds", None)

    return result


def run_inference(args) -> Path:
    fmt = _infer_format(args.output, args.format)

    inputs = args.input
    if not inputs:
        years = sorted({
            pd.Timestamp(args.period_start).year, pd.Timestamp(args.period_end).year
        })
        years = list(range(years[0], years[-1] + 1))
        inputs = resolve_paths(
            args.input_dir, args.input_template, years, experiment=args.experiment
        )
    for path in inputs:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    dataset = LazyXarraySourceOnlyDataset(
        source_fname=inputs,
        source_var=args.source_var,
        log_eps=args.log_eps,
        exclude_hours=args.exclude_hours,
        period_start=args.period_start,
        period_end=args.period_end,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No consecutive-hour pairs in {inputs} for "
            f"{args.period_start!r} to {args.period_end!r}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    _PredictModel.LOG_EPS = float(args.log_eps)
    _PredictModel.SUBTRACT_LOG_EPS = not args.keep_eps_in_total
    model = _PredictModel.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()

    accelerator = args.accelerator
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator, devices=args.devices,
        logger=False, enable_checkpointing=False,plugins=[LightningEnvironment()],
    )
    with torch.no_grad():
        batches = trainer.predict(model, dataloaders=loader)

    # predict_step returns linear units. The clamp is a guard.
    linear = torch.cat([b for b in batches if b is not None], dim=0).cpu().float()
    linear = torch.clamp(linear, min=0.0)

    n_start, n_lead, height, width = linear.shape
    if n_start != len(dataset):
        raise RuntimeError(
            f"{n_start} predictions for {len(dataset)} samples; alignment broken"
        )
    if n_lead != model.hparams.n_classes:
        raise RuntimeError(f"expected {model.hparams.n_classes} leads, got {n_lead}")

    step = 60 // n_lead
    first_centre = np.timedelta64(30 - step // 2, "m")
    lead_offsets = np.arange(0, 60, step).astype("timedelta64[m]")
    valid_times = (
        (dataset.sample_start_times - first_centre)[:, None] + lead_offsets[None, :]
    ).reshape(-1)

    source_ds = _canonical_source(inputs, args.source_var)
    result = build_output(
        linear.reshape(n_start * n_lead, height, width).numpy().astype(np.float32),
        valid_times, source_ds, args.source_var, n_lead, args.time_bounds,
    )

    bounded = set(result.coords) | {"time_bnds"}
    for name in list(result.coords) + [args.source_var, "time_bnds"]:
        if name in source_ds and name in result:
            encoding = _clean_encoding(source_ds[name].encoding, fmt)
            if name in bounded:                
                encoding["_FillValue"] = None
                encoding.pop("missing_value", None)
            result[name].encoding = encoding

    # Time and its bounds are written in minutes, and in the same
    # units: CF readers such as CDO interpret time_bnds using the units of the
    # time axis, so differing reference dates would shift the bounds.
    reference = pd.Timestamp(result["time_bnds"].values.min()).strftime("%Y-%m-%d %H:%M:%S")
    calendar = result["time"].encoding.get("calendar", "proleptic_gregorian")
    for name in ("time", "time_bnds"):
        result[name].encoding.update(
            {"units": f"minutes since {reference}", "calendar": calendar, "dtype": "int64"}
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "netcdf":
        result.to_netcdf(output)
    else:
        result.to_zarr(output, mode="w", consolidated=True)

    print(
        f"Wrote {output} ({fmt}): {n_start} hours -> "
        f"{n_start * n_lead} {step}-minute fields, "
        f"time_bnds convention '{args.time_bounds}'"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mldpit-infer",
        description="Downscale hourly precipitation to 10-minute precipitation.",
    )
    parser.add_argument("--checkpoint", required=True, help="Pretrained .ckpt file.")
    parser.add_argument("--output", required=True, help="Output .zarr or .nc path.")

    src = parser.add_argument_group("input (give --input, or the template trio)")
    src.add_argument("--input", nargs="*", default=None, help="Zarr stores to read.")
    src.add_argument("--input-dir", default=None)
    src.add_argument(
        "--input-template", default="train_source_{year}_{experiment}_1hr_hoursum.zarr"
    )
    src.add_argument("--experiment", default="")

    parser.add_argument("--period-start", default=None, help='e.g. "2020-01-01".')
    parser.add_argument("--period-end", default=None, help="Exclusive upper bound.")
    parser.add_argument("--source-var", default="TOT_PREC")
    parser.add_argument("--log-eps", type=float, default=1e-6)
    parser.add_argument("--exclude-hours", nargs="*", default=[])

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--accelerator", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--devices", type=int, default=1)

    parser.add_argument(
        "--time-bounds", default="legacy", choices=["legacy", "centred"],
        help="time_bnds convention: 'legacy' is [t-10min, t] as published; "
             "'centred' is [t-5min, t+5min], consistent with centre-stamped times.",
    )
    parser.add_argument(
        "--keep-eps-in-total", action="store_true",
        help="Distribute the hourly total including the log_eps offset, as the "
             "published output did. Dry cells then emerge at about 1e-9 kg m-2 "
             "rather than zero. Use only to reproduce the archived files.",
    )
    parser.add_argument(
        "--format", default=None, choices=["zarr", "netcdf"],
        help="Output format; inferred from the --output extension if omitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    if not args.input and not args.input_dir:
        raise SystemExit("Give either --input or --input-dir with --input-template.")
    run_inference(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
