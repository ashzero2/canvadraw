from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = (256, 256)  # both dims must be divisible by 8 for the u-net


class CanvaDataset(Dataset):
    def __init__(self, pairs_json: Path, sigma: float = 20.0) -> None:
        self.pairs = json.loads(pairs_json.read_text())
        self.root = pairs_json.parent.parent
        self.sigma = sigma
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]

        input_img = self.to_tensor(Image.open(self.root / pair["input"]).convert("RGB"))
        target_img = self.to_tensor(Image.open(self.root / pair["output"]).convert("RGB"))

        _, H, W = input_img.shape
        cx = int(pair["click_x"] * W / 400)  # scale click coords to resized dimensions
        cy = int(pair["click_y"] * H / 300)
        heatmap = self._gaussian_heatmap(cx, cy, H, W)

        return torch.cat([input_img, heatmap], dim=0), target_img

    def _gaussian_heatmap(self, x: int, y: int, H: int, W: int) -> torch.Tensor:
        grid_y = torch.arange(H, dtype=torch.float32).view(-1, 1).expand(H, W)
        grid_x = torch.arange(W, dtype=torch.float32).view(1, -1).expand(H, W)
        heatmap = torch.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * self.sigma**2))
        return heatmap.unsqueeze(0)
