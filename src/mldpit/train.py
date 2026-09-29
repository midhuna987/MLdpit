#Training for MLdpit

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from mldpit.config import parse_args_with_config, resolve_paths
from mldpit.datamodule import LazyDatamodule
from mldpit.model import LightningSRUNet
from mldpit.preprocess import parse_years


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mldpit-train",
        description="Train the temporal downscaling U-Net.",
    )

    data = parser.add_argument_group("data")
    data.add_argument("--source-dir", required=True, help="Directory of hourly Zarr stores.")
    data.add_argument("--target-dir", required=True, help="Directory of 10-minute Zarr stores.")
    data.add_argument(
        "--source-template", default="train_source_{year}_{experiment}_1hr_hoursum.zarr",
        help="Filename pattern for hourly stores; fields: {year}, {experiment}.",
    )
    data.add_argument(
        "--target-template", default="train_target_{year}_{experiment}_10mnts.zarr",
        help="Filename pattern for 10-minute stores; fields: {year}, {experiment}.",
    )
    data.add_argument("--experiment", default="", help="Experiment identifier.")
    data.add_argument("--train-years", required=True, help='e.g. "1960-2007,2012-2017".')
    data.add_argument("--val-years", required=True, help='e.g. "2018-2019".')
    data.add_argument(
        "--exclude-hours", nargs="*", default=[],
        help='Hours to drop, as "YYYY-MM-DDTHH".',
    )
    data.add_argument("--source-var", default="TOT_PREC")
    data.add_argument("--target-var", default="TOT_PREC")

    opt = parser.add_argument_group("optimisation")
    opt.add_argument("--epochs", type=int, default=10)
    opt.add_argument("--batch-size", type=int, default=4)
    opt.add_argument("--learning-rate", type=float, default=1e-4)
    opt.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    opt.add_argument("--accumulate-grad-batches", type=int, default=2)
    opt.add_argument("--early-stopping-patience", type=int, default=5)

    hw = parser.add_argument_group("hardware")
    hw.add_argument("--num-nodes", type=int, default=1)
    hw.add_argument("--devices", type=int, default=4)
    hw.add_argument("--num-workers", type=int, default=8)
    hw.add_argument("--precision", default="32-true")
    hw.add_argument("--strategy", default="ddp")
    hw.add_argument(
        "--accelerator", default="auto", choices=["auto", "gpu", "cpu"],
        help="'auto' selects gpu when CUDA is present, otherwise cpu.",
    )

    out = parser.add_argument_group("output")
    out.add_argument("--checkpoint-dir", default="checkpoints")
    out.add_argument("--log-dir", default="logs")
    out.add_argument("--resume", action="store_true", help="Resume from last.ckpt.")
    out.add_argument("--wandb-project", default=None, help="Enable W&B logging.")

    return parser


def build_loggers(args) -> list:
    loggers = [CSVLogger(save_dir=args.log_dir, name="csv")]
    if args.wandb_project:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError:
            print("wandb not installed; continuing with CSV logging only.",
                  file=sys.stderr)
        else:
            os.environ.setdefault("WANDB_SILENT", "true")
            loggers.append(WandbLogger(project=args.wandb_project, log_model="all"))
    return loggers


def main(argv: list[str] | None = None) -> int:
    args = parse_args_with_config(build_parser(), argv)
    torch.set_float32_matmul_precision("medium")
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    train_years = parse_years(args.train_years)
    val_years = parse_years(args.val_years)
    fields = {"experiment": args.experiment}

    datamodule = LazyDatamodule(
        train_source_fnames=resolve_paths(args.source_dir, args.source_template, train_years, **fields),
        train_target_fnames=resolve_paths(args.target_dir, args.target_template, train_years, **fields),
        val_source_fnames=resolve_paths(args.source_dir, args.source_template, val_years, **fields),
        val_target_fnames=resolve_paths(args.target_dir, args.target_template, val_years, **fields),
        source_var=args.source_var,
        target_var=args.target_var,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        exclude_hours=args.exclude_hours,
    )

    model = LightningSRUNet(lr=args.learning_rate, optimizer=args.optimizer)

    checkpoint_dir = Path(args.checkpoint_dir)
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir, filename="best_weight",
            monitor="val_loss", mode="min", save_top_k=1, save_last=True,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir, filename="checkpoint_epoch_{epoch:03d}",
            save_top_k=-1, every_n_epochs=1, auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val_loss", mode="min",
            patience=args.early_stopping_patience, verbose=True,
        ),
    ]

    ckpt_path = None
    if args.resume:
        candidate = checkpoint_dir / "last.ckpt"
        if candidate.exists():
            ckpt_path = str(candidate)
            print(f"Resuming from {ckpt_path}", flush=True)
        else:
            print("No last.ckpt found; starting from scratch.", flush=True)

    # The defaults (four devices, DDP, synchronised batch norm) suit Levante
    # GPU node, but cannot run on a machine without CUDA.
     
    accelerator = args.accelerator
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    strategy, devices, sync_batchnorm = args.strategy, args.devices, True
    if accelerator == "cpu":
        if (strategy, devices) != ("auto", 1):
            print(
                f"CPU accelerator: overriding --strategy {strategy!r} -> 'auto' "
                f"and --devices {devices} -> 1, and disabling sync_batchnorm.",
                flush=True,
            )
        strategy, devices, sync_batchnorm = "auto", 1, False

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        logger=build_loggers(args),
        callbacks=callbacks,
        precision=args.precision,
        strategy=strategy,
        devices=devices,
        accelerator=accelerator,
        num_nodes=args.num_nodes,
        log_every_n_steps=5,
        sync_batchnorm=sync_batchnorm,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
