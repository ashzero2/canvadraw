from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset.loader import CanvaDataset
from model.unet import UNet

EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-3
VAL_SPLIT = 0.1
CHECKPOINT = Path("canvadraw.pth")


def train() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = CanvaDataset(Path("dataset/pairs.json"))
    val_size = int(len(dataset) * VAL_SPLIT)
    train_ds, val_ds = random_split(
        dataset,
        [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    print(f"Train: {len(train_ds)} samples  Val: {len(val_ds)} samples")

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Non-white pixels (colored circles/text) get 10× weight so the model
        # can't escape by predicting a white mean — those ~4% of pixels dominate.
        mask = (target < 0.92).float().amax(dim=1, keepdim=True)
        weight = 1.0 + 9.0 * mask
        return (weight * (pred - target).abs()).mean()

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in val_dl:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item()
        val_loss /= len(val_dl)

        scheduler.step(val_loss)

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT)
            saved = "  ✓ saved"

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.0e}{saved}"
        )

    print(f"\nDone. Best val loss: {best_val_loss:.4f} → {CHECKPOINT}")


if __name__ == "__main__":
    train()
