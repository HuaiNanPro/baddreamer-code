#!/usr/bin/env python3
"""
Simple tokenize script for nuScenes dataset.
Uses CPU for tokenization.
"""
import os
import glob

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image

# Configuration
DATASET = "nuscenes"
RESIZE_FACTOR = 3.125

# Paths
BLOBS_DIR = "/raid/zengchaolv/sz/nuscenes/Nuscenes_trainval_v1.0/v1.0-trainval*_blobs/CAM_FRONT"
TOKENIZER_PATH = "/raid/zengchaolv/sz/vavim_finetune_nuscenes/VQ_ds16_16384_llamagen_encoder.jit"
OUTPUT_DIR = "/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/tokens"

def resize_by_factor(img, resize_factor):
    new_width = int(img.shape[2] / resize_factor)
    new_height = int(img.shape[1] / resize_factor)
    return TF.resize(img, (new_height, new_width), antialias=True)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tokenizer from " + TOKENIZER_PATH + " on CPU...")
    tokenizer = torch.jit.load(TOKENIZER_PATH, map_location='cpu').eval()

    print("Collecting image files...")
    image_files = sorted(glob.glob(BLOBS_DIR + "/*.jpg"))
    print("Found " + str(len(image_files)) + " images")

    batch_size = 32
    total = len(image_files)

    for i in range(0, total, batch_size):
        if i % 320 == 0:
            print("Processing " + str(i) + "/" + str(total))

        batch_files = image_files[i:i+batch_size]
        images = []

        for f in batch_files:
            frame = Image.open(f).convert("RGB")
            frame = TF.to_image(frame)
            frame = TF.to_dtype(frame, torch.uint8, scale=True)
            frame = resize_by_factor(frame, RESIZE_FACTOR)
            frame = TF.to_dtype(frame, torch.float32, scale=True)
            frame = 2.0 * frame - 1.0
            images.append(frame)

        batch_tensor = torch.stack(images)

        with torch.no_grad():
            tokens = tokenizer(batch_tensor)
            tokens = tokens.numpy()

        for j, f in enumerate(batch_files):
            basename = os.path.basename(f).replace('.jpg', '')
            token_path = os.path.join(OUTPUT_DIR, basename + ".npy")
            np.save(token_path, tokens[j])

    count = len(os.listdir(OUTPUT_DIR))
    print("Tokens saved to " + OUTPUT_DIR)
    print("Total tokens: " + str(count))

if __name__ == "__main__":
    main()