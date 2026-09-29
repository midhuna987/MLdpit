"""U-Net backbone for temporal downscaling of precipitation fields.

 The architecture is the four-level U-Net of Ronneberger et al. (2015) with batch
 normalisation added to each convolution block. Spatial dimensions are handled by 
 reflect-padding the input up to the next multiple of 16 and cropping the output back 
 to the original input grid, so the module accepts any grid rather than only the 501 x 501 ICON 
 domain it was trained on.  For a 501 x 501 input this reproduces exactly the (5, 6, 5, 6)
 padding and [5:506, 5:506] crop used during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => batch norm => ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Four-level U-Net.
    
    n_channels:
        Number of input channels.  Two in the default configuration: the hourly
        precipitation total of the target hour and of the following hour.
    n_classes:
        Number of output channels, i.e. the number of sub-hourly intervals.
        Six in the default 10-minute configuration.
    """

    # Inputs are padded up to a multiple of this value so that four successive
    # 2x max-pooling operations divide evenly.
    SIZE_MULTIPLE = 16

    def __init__(self, n_channels: int = 2, n_classes: int = 6) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def _pad(self, x: torch.Tensor):
        """Reflect-pad ``x`` up to a multiple of :attr:`SIZE_MULTIPLE`."""
        height, width = x.shape[-2:]
        pad_h = (self.SIZE_MULTIPLE - height % self.SIZE_MULTIPLE) % self.SIZE_MULTIPLE
        pad_w = (self.SIZE_MULTIPLE - width % self.SIZE_MULTIPLE) % self.SIZE_MULTIPLE
        top, left = pad_h // 2, pad_w // 2
        padded = F.pad(x, (left, pad_w - left, top, pad_h - top), mode="reflect")
        return padded, (top, height, left, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (top, height, left, width) = self._pad(x)

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder, with skip connections
        x = self.conv1(torch.cat([x4, self.up1(x5)], dim=1))
        x = self.conv2(torch.cat([x3, self.up2(x)], dim=1))
        x = self.conv3(torch.cat([x2, self.up3(x)], dim=1))
        x = self.conv4(torch.cat([x1, self.up4(x)], dim=1))
        x = self.outc(x)

        return x[:, :, top : top + height, left : left + width]
