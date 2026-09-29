"""Datasets pairing hourly source fields with sub-hourly target fields.

Timestamp convention
--------------------
Both source and target timestamps denote **interval centres**.  An hourly field
covering 00:00-01:00 is stamped 00:30; the six 10-minute fields covering that
same hour are stamped 00:05, 00:15, ... 00:55.  A training sample therefore
consists of the hourly fields stamped ``HH:30`` and ``(HH+1):30`` as the two
input channels, and the six 10-minute fields within hour ``HH`` as the target.

The second hourly field is supplied as context for the temporal evolution
across the hour boundary; only the first is used as the total that the
mass-conservation constraint must reproduce.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import dask
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

#: Variable names accepted in the input files, harmonised to a single name.
VAR_CANDIDATES = ("TOT_PREC", "TOT_PREC_10min")


def sub_hourly_offsets(n_sub_intervals: int = 6) -> tuple[str, ...]:
    
    if 60 % n_sub_intervals:
        raise ValueError(f"{n_sub_intervals} sub-intervals do not divide an hour")
    step = 60 // n_sub_intervals
    return tuple(f":{minute:02d}:00" for minute in range(step // 2, 60, step))


def _harmoniser(canonical: str, candidates: Sequence[str] = VAR_CANDIDATES):    

    def _preprocess(ds: xr.Dataset) -> xr.Dataset:
        for name in candidates:
            if name in ds.data_vars:
                return ds.rename({name: canonical}) if name != canonical else ds
        raise KeyError(f"none of {tuple(candidates)} in {list(ds.data_vars)}")

    return _preprocess


def _open(fnames, var: str) -> xr.Dataset:
    return xr.open_mfdataset(
        fnames, combine="nested", concat_dim="time",
        engine="zarr", preprocess=_harmoniser(var),
    )


class LazyXarrayDataset(Dataset):    

    def __init__(
        self,
        source_fname,
        target_fname,
        source_var: str = "TOT_PREC",
        target_var: str = "TOT_PREC",
        log_eps: float = 1e-6,
        exclude_hours: Iterable[str] | None = None,
        n_sub_intervals: int = 6,
        var_candidates: Sequence[str] = VAR_CANDIDATES,
    ) -> None:
        dask.config.set(scheduler="synchronous")

        self.source_ds = _open(source_fname, source_var)
        self.target_ds = _open(target_fname, target_var)
        self.source = self.source_ds[source_var]
        self.target = self.target_ds[target_var]
        self.log_eps = log_eps
        self.n_sub_intervals = n_sub_intervals

        offsets = sub_hourly_offsets(n_sub_intervals)

        self.exclude_src: set[np.datetime64] = set()
        self.exclude_tgt: set[np.datetime64] = set()
        for hour in exclude_hours or ():
            self.exclude_src.add(np.datetime64(f"{hour}:30:00", "ns"))
            for offset in offsets:
                self.exclude_tgt.add(np.datetime64(hour + offset, "ns"))

        source_times = {np.datetime64(t, "ns") for t in self.source["time"].values}
        target_times = {np.datetime64(t, "ns") for t in self.target["time"].values}

        self.samples: list[tuple] = []
        for raw in self.source["time"].values:
            t0 = np.datetime64(raw, "ns")
            if t0 in self.exclude_src:
                continue
            t1 = t0 + np.timedelta64(1, "h")
            if t1 not in source_times or t1 in self.exclude_src:
                continue
            hour_key = str(t0)[:13]  # "YYYY-MM-DDTHH"
            targets = [np.datetime64(hour_key + o, "ns") for o in offsets]
            if all(t in target_times and t not in self.exclude_tgt for t in targets):
                self.samples.append((t0, t1, targets))

        if not self.samples:
            raise RuntimeError(
                "No source/target pairs found. Check that the source and target "
                "Zarr stores cover the same period and use the expected "
                "timestamp convention (hourly at HH:30, sub-hourly at centres)."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        t0, t1, targets = self.samples[idx]

        x = np.stack([
            self.source.sel(time=t0).astype(np.float32).values,
            self.source.sel(time=t1).astype(np.float32).values,
        ], axis=0)
        y = np.stack([
            self.target.sel(time=t).astype(np.float32).values for t in targets
        ], axis=0)

        x = np.log10(x + self.log_eps)
        y = np.log10(y + self.log_eps)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class LazyXarraySourceOnlyDataset(Dataset):
    
    def __init__(
        self,
        source_fname,
        source_var: str = "TOT_PREC",
        log_eps: float = 1e-6,
        exclude_hours: Iterable[str] | None = None,
        period_start: str | None = None,
        period_end: str | None = None,
        var_candidates: Sequence[str] = VAR_CANDIDATES,
    ) -> None:
        dask.config.set(scheduler="synchronous")

        self.source_ds = _open(source_fname, source_var)
        self.source = self.source_ds[source_var]
        self.log_eps = log_eps

        self.exclude_src = {
            np.datetime64(f"{hour}:30:00", "ns") for hour in (exclude_hours or ())
        }

        source_times = {np.datetime64(t, "ns") for t in self.source["time"].values}
        start = np.datetime64(period_start, "ns") if period_start else None
        end = np.datetime64(period_end, "ns") if period_end else None

        self.samples: list[tuple] = []
        for raw in self.source["time"].values:
            t0 = np.datetime64(raw, "ns")
            if start is not None and t0 < start:
                continue
            if end is not None and t0 >= end:
                continue
            if t0 in self.exclude_src:
                continue
            t1 = t0 + np.timedelta64(1, "h")
            if t1 not in source_times or t1 in self.exclude_src:
                continue
            self.samples.append((t0, t1))

        self.sample_start_times = np.array(
            [s[0] for s in self.samples], dtype="datetime64[ns]"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        t0, t1 = self.samples[idx]
        x = np.stack([
            self.source.sel(time=t0).astype(np.float32).values,
            self.source.sel(time=t1).astype(np.float32).values,
        ], axis=0)
        return torch.from_numpy(np.log10(x + self.log_eps)).float()
