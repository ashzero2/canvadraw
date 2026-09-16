from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _conv_bn_relu(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))


class DecoderBlock(nn.Module):
    """Upsample → conv, no skip connections — pure generation path."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _conv_bn_relu(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class UNet(nn.Module):
    """Encoder-decoder with FiLM label conditioning at the bottleneck.

    Skip connections are intentionally absent: this is conditional generation
    (login-form → welcome/error screen), not reconstruction. Skip connections
    would push the decoder toward reproducing the input instead of generating
    the target output class.

    FiLM injects the label (0=error, 1=welcome) by scaling and shifting the
    256-channel bottleneck features, ensuring the decoder always knows which
    output to produce regardless of the input image content.
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 3) -> None:
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 32)  # 256→128
        self.enc2 = EncoderBlock(32, 64)  # 128→64
        self.enc3 = EncoderBlock(64, 128)  # 64→32

        self.bottleneck = _conv_bn_relu(128, 256)  # 32×32

        # FiLM: label scalar → scale + shift for all 256 bottleneck channels
        self.film_scale = nn.Linear(1, 256)
        self.film_bias = nn.Linear(1, 256)

        self.dec3 = DecoderBlock(256, 128)  # 32→64
        self.dec2 = DecoderBlock(128, 64)  # 64→128
        self.dec1 = DecoderBlock(64, 32)  # 128→256

        self.out = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Init FiLM scale near 1, bias near 0 so early training is stable
        nn.init.normal_(self.film_scale.weight, std=0.02)
        nn.init.ones_(self.film_scale.bias)
        nn.init.normal_(self.film_bias.weight, std=0.02)
        nn.init.zeros_(self.film_bias.bias)

    def _film(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        gamma = self.film_scale(label).view(-1, 256, 1, 1)
        beta = self.film_bias(label).view(-1, 256, 1, 1)
        return gamma * z + beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        label = x[:, 4:5, 0, 0]  # uniform class_channel → scalar per sample
        z = self.enc1(x)
        z = self.enc2(z)
        z = self.enc3(z)
        z = self.bottleneck(z)
        z = self._film(z, label)
        z = self.dec3(z)
        z = self.dec2(z)
        z = self.dec1(z)
        return self.out(z)

    def forward_with_steps(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (step1, step2, final) — decoder activation heatmaps + output."""
        label = x[:, 4:5, 0, 0]
        z = self.enc1(x)
        z = self.enc2(z)
        z = self.enc3(z)
        z = self.bottleneck(z)
        z = self._film(z, label)

        z = self.dec3(z)
        step1 = self._heatmap(z)  # [B, 3, 64, 64]

        z = self.dec2(z)
        step2 = self._heatmap(z)  # [B, 3, 128, 128]

        z = self.dec1(z)
        return step1, step2, self.out(z)  # final: [B, 3, 256, 256]

    @staticmethod
    def _heatmap(x: torch.Tensor) -> torch.Tensor:
        """Mean activation magnitude across channels → normalized grayscale RGB."""
        mag = x.abs().mean(dim=1, keepdim=True)
        lo = mag.amin(dim=(2, 3), keepdim=True)
        hi = mag.amax(dim=(2, 3), keepdim=True)
        norm = (mag - lo) / (hi - lo + 1e-8)
        return norm.expand(-1, 3, -1, -1)
