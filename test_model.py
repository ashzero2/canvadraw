"""
Tests model generalisation by running inference on freshly drawn images
that were never part of the training dataset.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from dataset.generate import draw_login
from dataset.loader import IMAGE_SIZE, CanvaDataset
from model.classifier import Classifier
from model.unet import UNet

CHECKPOINT = Path("canvadraw.pth")
CLASSIFIER_CHECKPOINT = Path("classifier.pth")


class ModelShapeTests(unittest.TestCase):
    def test_unet_output_shape(self) -> None:
        model = UNet().eval()
        x = torch.rand(1, 5, *IMAGE_SIZE)

        with torch.no_grad():
            output = model(x)

        self.assertEqual(output.shape, (1, 3, *IMAGE_SIZE))

    def test_classifier_output_shape(self) -> None:
        model = Classifier().eval()
        x = torch.rand(2, 3, *IMAGE_SIZE)

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (2,))


def load_models(device: torch.device) -> tuple[UNet, Classifier]:
    unet = UNet().to(device)
    unet.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    unet.eval()
    classifier = Classifier().to(device)
    classifier.load_state_dict(
        torch.load(CLASSIFIER_CHECKPOINT, map_location=device, weights_only=True)
    )
    classifier.eval()
    return unet, classifier


def infer(
    model: UNet, classifier: Classifier, img: Image.Image, device: torch.device
) -> Image.Image:
    to_tensor = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
    inp = to_tensor(img.convert("RGB"))

    # click heatmap at submit button center (scaled to 256x256)
    dataset = CanvaDataset(Path("dataset/pairs.json"))
    cx = int(200 * IMAGE_SIZE[1] / 400)
    cy = int(210 * IMAGE_SIZE[0] / 300)
    heatmap = dataset._gaussian_heatmap(cx, cy, IMAGE_SIZE[0], IMAGE_SIZE[1])

    with torch.no_grad():
        label = float(classifier(inp.unsqueeze(0).to(device)).sigmoid().item() >= 0.5)
        class_channel = torch.full((1, *IMAGE_SIZE), label)
        x = torch.cat([inp, heatmap, class_channel], dim=0).unsqueeze(0).to(device)
        pred = model(x).squeeze(0).cpu()
    return to_pil_image(pred)


def run_tests() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, classifier = load_models(device)

    # usernames the model has never seen before
    test_cases = [
        ("rahul", "in training → welcome"),  # seen in training
        ("bob", "never seen → should welcome?"),  # new, looks like a valid name
        ("hacker", "in training → error"),  # seen in training
        ("evil", "never seen → should error?"),  # new, unknown to model
        ("admin", "in training → welcome"),  # seen in training
        ("unknown", "never seen → ?"),  # completely new
    ]

    panels = []
    for username, description in test_cases:
        login_img = draw_login(username)
        pred_img = infer(model, classifier, login_img, device)

        W, H = IMAGE_SIZE
        row = Image.new("RGB", (W * 2 + 10, H), color=(200, 200, 200))
        row.paste(login_img.resize((W, H)), (0, 0))
        row.paste(pred_img, (W + 10, 0))
        panels.append((row, username, description))
        print(f"  {username:10s}  ({description})")

    total_h = sum(p[0].height + 5 for p in panels) + 30
    out = Image.new("RGB", (panels[0][0].width, total_h), color=(240, 240, 240))
    y = 15
    for row, _, _ in panels:
        out.paste(row, (0, y))
        y += row.height + 5

    out.save("test_results.png")
    print("\nSaved: test_results.png  [login form | prediction]")
    print("Green tint = model thinks welcome, Red/pink tint = model thinks error")


if __name__ == "__main__":
    run_tests()
