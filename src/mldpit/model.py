"""LightningModule wrapping the U-Net with a mass-conservation constraint.
   
    A softmax over "dim" turns the raw network output into weights summing to
    one across the sub-hourly intervals; multiplying by the hourly total makes
    the six 10-minute fields sum exactly to the hourly field they came from.
    Mass conservation is therefore structural, not a soft penalty in the loss.
    
    """

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from mldpit.unet import UNet


def softmax_constraint(
    raw_linear: torch.Tensor,
    hour_total_linear: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:    
    weights = F.softmax(raw_linear, dim=dim)
    return torch.clamp(weights * hour_total_linear, min=0.0)


class LightningSRUNet(pl.LightningModule):    

    def __init__(
        self,
        lr: float = 1e-4,
        optimizer: str = "adamw",
        n_channels: int = 2,
        n_classes: int = 6,
        log_eps: float = 1e-6,
        lr_patience: int = 2,
        lr_factor: float = 0.1,
    ) -> None:
        super().__init__()
        # Records the constructor arguments in the checkpoint so that
        # load_from_checkpoint restores what was trained rather than silently
        # falling back to these defaults.
        self.save_hyperparameters()

        self.model = UNet(n_channels=n_channels, n_classes=n_classes)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor, hour_total_linear: torch.Tensor) -> torch.Tensor:
        raw_log = self.model(x)
        raw_linear = 10.0**raw_log
        constrained_linear = softmax_constraint(raw_linear, hour_total_linear, dim=1)
        return torch.log10(constrained_linear + self.hparams.log_eps)

    @staticmethod
    def hourly_total(x: torch.Tensor) -> torch.Tensor:
        """Hourly total of the target hour, in linear units."""
        return (10.0**x)[:, 0:1, ...]

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch  # both in log10 space
        loss = self.criterion(self(x, self.hourly_total(x)), y)
        self.log(
            f"{stage}_loss", loss,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x, self.hourly_total(x))

    def configure_optimizers(self):
        name = self.hparams.optimizer.lower()
        if name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        elif name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer!r}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            patience=self.hparams.lr_patience, factor=self.hparams.lr_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
