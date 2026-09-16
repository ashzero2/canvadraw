from __future__ import annotations

import torch
import torch.nn as nn


class Classifier(nn.Module):
    """Binary CNN classifier: canvas image → welcome/error logit."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grayscale = x.mean(dim=1, keepdim=True)
        username = grayscale[:, :, 98:126, 32:224]
        password = grayscale[:, :, 145:173, 32:224]
        fields = torch.cat([username, password], dim=2)
        return self.head(fields).squeeze(1)
