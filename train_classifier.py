from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset.loader import CanvaDataset
from model.classifier import Classifier

EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
CHECKPOINT = Path("classifier.pth")


def train() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training classifier on: {device}")

    dataset = CanvaDataset(Path("dataset/pairs.json"))
    val_size = int(len(dataset) * 0.1)
    train_ds, val_ds = random_split(
        dataset,
        [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    model = Classifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for x, _, label in train_dl:
            x, label = x[:, :3].to(device), label.to(device)  # classifier only needs RGB
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += ((logits > 0) == label.bool()).sum().item()
            total += len(label)
        train_loss /= len(train_dl)
        train_acc = correct / total

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, _, label in val_dl:
                x, label = x[:, :3].to(device), label.to(device)
                logits = model(x)
                val_loss += loss_fn(logits, label).item()
                correct += ((logits > 0) == label.bool()).sum().item()
                total += len(label)
        val_loss /= len(val_dl)
        val_acc = correct / total

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT)
            saved = "  ✓ saved"

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"train={train_loss:.4f} acc={train_acc:.2%}  "
            f"val={val_loss:.4f} acc={val_acc:.2%}{saved}"
        )

    print(f"\nDone. Best val loss: {best_val_loss:.4f} → {CHECKPOINT}")


if __name__ == "__main__":
    train()
