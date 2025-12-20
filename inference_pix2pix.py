#!/usr/bin/env python3
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import argparse

# ===== Import your ResnetGenerator model here =====
from train_pix2pix import ResnetGenerator
from generate_data import make_skeleton_from_bitmap, dilate_mask_to_black_strokes

# def load_image_as_tensor(path, size=512):
#     """Loads skeleton image and converts to normalized tensor shape (1,3,H,W) in [-1,1]."""
#     img = Image.open(path).convert("L").resize((size, size))
#     img = np.array(img)
#     Image.fromarray(img).save("debug_input.png")
#     sk = make_skeleton_from_bitmap(img)
#     sk = dilate_mask_to_black_strokes(sk, kernel_size=5)
#     Image.fromarray(sk).save("debug_skeleton.png")
#       # save for debugging
#     # arr = np.array(img).astype(np.float32) / 255.0
#     arr = sk.astype(np.float32) / 255.0
#     arr = arr * 2.0 - 1.0  # map [0,1] → [-1,1]
#     arr = arr.transpose(2, 0, 1)  # HWC → CHW
#     tensor = torch.tensor(arr).unsqueeze(0)  # add batch dim
#     return tensor

def load_image_as_tensor(path, size=512):
    """Loads skeleton image and converts to normalized tensor shape (1,3,H,W) in [-1,1]."""
    # --- Load grayscale ---
    img = Image.open(path).convert("L").resize((size, size))
    img = np.array(img)

    # Image.fromarray(img).save("debug_input.png")

    # --- Skeletonize ---
    sk = make_skeleton_from_bitmap(img)
    # Image.fromarray(sk).save("debug_skeleton_before_dilate.png")
    sk = dilate_mask_to_black_strokes(sk, kernel_size=3)

    # Save debug output
    # Image.fromarray(sk).save("debug_skeleton.png")

    # --- Convert skeleton (H,W) -> (H,W,3) ---
    sk_rgb = np.stack([sk, sk, sk], axis=-1)

    # --- Normalize to [-1,1] ---
    arr = sk_rgb.astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # [0,1] → [-1,1]

    # --- HWC → CHW ---
    arr = arr.transpose(2, 0, 1)

    # --- Add batch dim: (1,3,H,W) ---
    tensor = torch.tensor(arr).unsqueeze(0)

    return tensor


def tensor_to_image(tensor):
    """Converts output tensor (1,3,H,W) in [-1,1] → uint8 RGB HWC"""
    img = tensor.detach().cpu().numpy()[0]
    img = (img + 1.0) / 2.0  # back to [0,1]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    return Image.fromarray(img)


@torch.no_grad()
def run_inference(ckpt_path, input_image, output_path, img_size=512, ngf=64):
    ckpt_path = Path(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # build generator
    G = ResnetGenerator(
        in_channels=3,
        out_channels=3,
        ngf=ngf,
        n_blocks=9,
    ).to(device)

    # load checkpoint
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    # load input skeleton
    A = load_image_as_tensor(input_image, size=img_size).to(device)

    # forward
    fake_B = G(A)

    # convert & save
    img_out = tensor_to_image(fake_B)
    output_path = Path(output_path)
    img_out.save(output_path)
    print(f"Saved generated image to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--input", required=True, help="Path to skeleton image")
    parser.add_argument("--out", required=True, help="Output path for generated png")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--ngf", type=int, default=64)
    args = parser.parse_args()

    run_inference(args.ckpt, args.input, args.out, args.size, args.ngf)
