#LightningDataModule for paired hourly/sub-hourly precipitation.

from __future__ import annotations

from typing import Iterable

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from mldpit.dataset import LazyXarrayDataset


class LazyDatamodule(LightningDataModule):
    """Train and validation loaders over lazily opened Zarr stores."""

    def __init__(
        self,
        train_source_fnames,
        train_target_fnames,
        val_source_fnames,
        val_target_fnames,
        source_var: str = "TOT_PREC",
        target_var: str = "TOT_PREC",
        batch_size: int = 4,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 1,
        exclude_hours: Iterable[str] | None = None,
        n_sub_intervals: int = 6,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=[
            "train_source_fnames", "train_target_fnames",
            "val_source_fnames", "val_target_fnames",
        ])
        self.train_source_fnames = train_source_fnames
        self.train_target_fnames = train_target_fnames
        self.val_source_fnames = val_source_fnames
        self.val_target_fnames = val_target_fnames

    def setup(self, stage: str | None = None) -> None:
        if stage not in ("fit", None):
            return
        h = self.hparams
        common = dict(
            source_var=h.source_var, target_var=h.target_var,
            exclude_hours=h.exclude_hours, n_sub_intervals=h.n_sub_intervals,
        )
        self.train_dataset = LazyXarrayDataset(
            self.train_source_fnames, self.train_target_fnames, **common
        )
        self.val_dataset = LazyXarrayDataset(
            self.val_source_fnames, self.val_target_fnames, **common
        )

    def _loader(self, dataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        h = self.hparams
        return DataLoader(
            dataset,
            batch_size=h.batch_size,
            shuffle=shuffle,
            num_workers=h.num_workers,
            pin_memory=h.pin_memory,
            persistent_workers=h.persistent_workers if h.num_workers > 0 else False,
            prefetch_factor=h.prefetch_factor if h.num_workers > 0 else None,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False, drop_last=False)
