from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from dataset.generate import draw_login
from dataset.loader import IMAGE_SIZE
from model.unet import UNet

VALID_USERS = {"admin", "rahul", "user"}
CANVAS_W, CANVAS_H = 400, 300

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
unet = UNet().to(device)
unet.load_state_dict(torch.load(Path("canvadraw.pth"), map_location=device, weights_only=True))
unet.eval()
print(f"Model loaded on {device}")

app = FastAPI()

to_tensor = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])


def gaussian_heatmap(x: int, y: int, H: int, W: int, sigma: float = 20.0) -> torch.Tensor:
    grid_y = torch.arange(H, dtype=torch.float32).view(-1, 1).expand(H, W)
    grid_x = torch.arange(W, dtype=torch.float32).view(1, -1).expand(H, W)
    heatmap = torch.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma**2))
    return heatmap.unsqueeze(0)


def tensor_to_b64(t: torch.Tensor, size: tuple[int, int] | None = None) -> str:
    img = to_pil_image(t.squeeze(0).clamp(0, 1))
    if size:
        img = img.resize(size, Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.websocket("/ws")
async def ws_predict(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()

            username: str = data["username"]
            click_x: int = data["click_x"]
            click_y: int = data["click_y"]

            label = 1.0 if username.strip().lower() in VALID_USERS else 0.0

            # Use the same Pillow renderer as training — eliminates browser-canvas domain shift.
            H, W = IMAGE_SIZE
            pillow_img = draw_login(username.strip())
            rgb = to_tensor(pillow_img)
            cx = int(click_x * W / CANVAS_W)
            cy = int(click_y * H / CANVAS_H)
            heatmap = gaussian_heatmap(cx, cy, H, W)
            class_ch = torch.full((1, H, W), label)
            x = torch.cat([rgb, heatmap, class_ch], dim=0).unsqueeze(0).to(device)

            t0 = time.perf_counter()

            with torch.no_grad():
                step1, step2, final = unet.forward_with_steps(x)

            elapsed_ms = round((time.perf_counter() - t0) * 1000)

            await websocket.send_json(
                {
                    "type": "step",
                    "step": 1,
                    "shape": list(step1.shape),
                    "image": tensor_to_b64(step1, size=(192, 144)),
                }
            )

            await websocket.send_json(
                {
                    "type": "step",
                    "step": 2,
                    "shape": list(step2.shape),
                    "image": tensor_to_b64(step2, size=(192, 144)),
                }
            )

            await websocket.send_json(
                {
                    "type": "done",
                    "image": tensor_to_b64(final, size=(CANVAS_W, CANVAS_H)),
                    "label": "welcome" if label == 1.0 else "error",
                    "inference_ms": elapsed_ms,
                    "input_shape": list(x.shape),
                    "username": username,
                    "click": {"x": click_x, "y": click_y},
                }
            )

    except WebSocketDisconnect:
        pass


@app.get("/")
async def root() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
