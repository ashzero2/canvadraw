"""
Run the full classifier → U-Net inference pipeline on a dataset sample.
Shows: input | expected output | predicted output side by side.

Usage: python predict.py
Saves: prediction.png
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from dataset.loader import CanvaDataset
from model.classifier import Classifier
from model.unet import UNet

UNET_CHECKPOINT = Path("canvadraw.pth")
CLF_CHECKPOINT = Path("classifier.pth")


def predict(sample_idx: int = 0) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    unet = UNet().to(device)
    unet.load_state_dict(torch.load(UNET_CHECKPOINT, map_location=device, weights_only=True))
    unet.eval()

    clf = Classifier().to(device)
    clf.load_state_dict(torch.load(CLF_CHECKPOINT, map_location=device, weights_only=True))
    clf.eval()

    dataset = CanvaDataset(Path("dataset/pairs.json"))
    inp, target, gt_label = dataset[sample_idx]

    with torch.no_grad():
        rgb = inp[:3].unsqueeze(0).to(device)
        welcome_probability = clf(rgb).sigmoid().item()
        predicted_label = float(welcome_probability >= 0.5)
        class_channel = torch.full((1, 1, *inp.shape[1:]), predicted_label, device=device)

        # rebuild input with classifier's predicted class channel
        x = torch.cat([rgb, inp[3:4].unsqueeze(0).to(device), class_channel], dim=1)
        predicted = unet(x).squeeze(0).cpu()

    input_img = to_pil_image(inp[:3])
    target_img = to_pil_image(target)
    pred_img = to_pil_image(predicted)

    W, H = input_img.size
    comparison = Image.new("RGB", (W * 3, H))
    comparison.paste(input_img, (0, 0))
    comparison.paste(target_img, (W, 0))
    comparison.paste(pred_img, (W * 2, 0))
    comparison.save("prediction.png")

    label_str = "welcome" if gt_label.item() == 1.0 else "error"
    clf_str = "welcome" if predicted_label == 1.0 else "error"
    print(f"Sample {sample_idx}  gt={label_str}  classifier={clf_str} ({welcome_probability:.2f})")
    print("Saved: prediction.png  [input | expected | predicted]")


if __name__ == "__main__":
    predict(sample_idx=0)
