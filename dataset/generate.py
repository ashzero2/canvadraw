"""
Generates synthetic training pairs for CanvaDraw.

Each pair = (input image, click_x, click_y, output image)
- Input:  login form with some username filled in
- Output: welcome screen (correct creds) or error screen (wrong creds)

Run: python dataset/generate.py
Outputs ~600 image pairs into dataset/images/
Also writes dataset/pairs.json listing every pair.
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H = 400, 300
OUT_DIR = Path(__file__).parent / "images"
OUT_DIR.mkdir(exist_ok=True)

CORRECT_USERS = ["admin", "rahul", "user"]
WRONG_USERS = ["guest", "hacker", "test", "foo", "root"]
PASSWORD = "1234"  # always the same for simplicity
WRONG_PASSWORDS = ["", "12", "123", "password", "0000"]

# Submit button region — center is where click will land
SUBMIT_X1, SUBMIT_Y1 = 140, 195
SUBMIT_X2, SUBMIT_Y2 = 260, 225
SUBMIT_CX = (SUBMIT_X1 + SUBMIT_X2) // 2
SUBMIT_CY = (SUBMIT_Y1 + SUBMIT_Y2) // 2


def draw_login(username: str, password: str = PASSWORD, noise: float = 0.0) -> Image.Image:
    img = Image.new("RGB", (W, H), color=(245, 245, 245))
    d = ImageDraw.Draw(img)

    # Card background
    d.rounded_rectangle(
        [30, 40, 370, 265], radius=8, fill=(255, 255, 255), outline=(210, 210, 210), width=1
    )

    # Title
    d.text((W // 2, 65), "Login", fill=(30, 30, 30), anchor="mm")

    # Username label + field
    d.text((50, 100), "Username", fill=(90, 90, 90))
    d.rounded_rectangle(
        [50, 115, 350, 145], radius=4, fill=(255, 255, 255), outline=(180, 180, 180), width=1
    )
    d.text((60, 123), username, fill=(20, 20, 20))

    # Password label + field
    d.text((50, 155), "Password", fill=(90, 90, 90))
    d.rounded_rectangle(
        [50, 170, 350, 200], radius=4, fill=(255, 255, 255), outline=(180, 180, 180), width=1
    )
    d.text((60, 178), "•" * len(password), fill=(20, 20, 20))

    # Submit button
    d.rounded_rectangle([SUBMIT_X1, SUBMIT_Y1, SUBMIT_X2, SUBMIT_Y2], radius=4, fill=(59, 130, 246))
    d.text((SUBMIT_CX, SUBMIT_CY), "Submit", fill=(255, 255, 255), anchor="mm")

    if noise > 0:
        arr = np.array(img, dtype=np.float32)
        arr += np.random.normal(0, noise * 255, arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def draw_welcome(username: str) -> Image.Image:
    img = Image.new("RGB", (W, H), color=(240, 253, 244))  # light green bg
    d = ImageDraw.Draw(img)

    d.rounded_rectangle(
        [30, 80, 370, 220], radius=8, fill=(255, 255, 255), outline=(134, 239, 172), width=2
    )

    # Checkmark circle
    d.ellipse([170, 95, 230, 155], fill=(34, 197, 94))
    d.text((200, 125), "✓", fill=(255, 255, 255), anchor="mm")

    d.text((W // 2, 170), f"Welcome, {username}!", fill=(21, 128, 61), anchor="mm")
    d.text((W // 2, 195), "Login successful", fill=(74, 222, 128), anchor="mm")

    return img


def draw_error() -> Image.Image:
    img = Image.new("RGB", (W, H), color=(254, 242, 242))  # light red bg
    d = ImageDraw.Draw(img)

    d.rounded_rectangle(
        [30, 80, 370, 220], radius=8, fill=(255, 255, 255), outline=(252, 165, 165), width=2
    )

    # X circle
    d.ellipse([170, 95, 230, 155], fill=(239, 68, 68))
    d.text((200, 125), "✕", fill=(255, 255, 255), anchor="mm")

    d.text((W // 2, 170), "Access Denied", fill=(185, 28, 28), anchor="mm")
    d.text((W // 2, 195), "Invalid credentials", fill=(248, 113, 113), anchor="mm")

    return img


def generate(n_per_class: int = 300) -> None:
    pairs = []
    idx = 0

    def save_pair(
        input_img: Image.Image,
        output_img: Image.Image,
        label: str,
        username: str,
        password: str,
    ) -> None:
        nonlocal idx
        in_path = OUT_DIR / f"{idx:04d}_input.png"
        out_path = OUT_DIR / f"{idx:04d}_output.png"
        input_img.save(in_path)
        output_img.save(out_path)
        pairs.append(
            {
                "input": str(in_path.relative_to(Path(__file__).parent.parent)),
                "output": str(out_path.relative_to(Path(__file__).parent.parent)),
                "click_x": SUBMIT_CX,
                "click_y": SUBMIT_CY,
                "label": label,  # "welcome" or "error"
                "username": username,
                "password": password,
            }
        )
        idx += 1

    # Correct credentials → welcome screen
    for _ in range(n_per_class):
        user = random.choice(CORRECT_USERS)
        noise = random.uniform(0, 0.03)  # tiny noise for augmentation
        save_pair(draw_login(user, PASSWORD, noise), draw_welcome(user), "welcome", user, PASSWORD)

    # Wrong credentials → error screen
    for _ in range(n_per_class):
        valid_username = random.random() < 0.35
        user = random.choice(CORRECT_USERS if valid_username else WRONG_USERS)
        password = random.choice(WRONG_PASSWORDS) if valid_username else PASSWORD
        noise = random.uniform(0, 0.03)
        save_pair(draw_login(user, password, noise), draw_error(), "error", user, password)

    pairs_file = Path(__file__).parent / "pairs.json"
    pairs_file.write_text(json.dumps(pairs, indent=2))

    print(f"Generated {len(pairs)} pairs → {OUT_DIR}")
    print(f"  {n_per_class} welcome  |  {n_per_class} error")
    print(f"  Canvas size: {W}x{H}  |  Click: ({SUBMIT_CX}, {SUBMIT_CY})")
    print(f"  Saved index → {pairs_file}")


if __name__ == "__main__":
    generate()
