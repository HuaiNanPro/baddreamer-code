"""
Convert Poisoned Images to VaViM Token Format
=====================================

Convert the 5% poisoned dataset images to the token format expected by VaViM.

The poisoned data structure:
  /raid/zengchaolv/shuaizhe_vavam/poisoned_5%/
    v1.0-trainval01_blobs/CAM_FRONT/*.jpg
    ...
    v1.0-trainval10_blobs/CAM_FRONT/*.jpg

Output format (matching OpenDV):
  /path/to/output/
    video_id_001/
      t_000000.npy
      t_000001.npy
      ...
    video_id_002/
      ...
"""

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import click
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_filename(filename: str) -> Tuple[str, int]:
    """
    Parse filename to extract video_id and frame_num.

    Filename format: n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151061512404.jpg
    Returns: (video_id, frame_num)
    """
    # Remove extension
    stem = Path(filename).stem

    # Split by __CAM_FRONT__
    parts = stem.split("__CAM_FRONT__")
    if len(parts) != 2:
        return None, None

    video_id = parts[0]  # e.g., n008-2018-08-01-15-16-36-0400
    frame_timestamp = parts[1]  # e.g., 1533151061512404

    return video_id, int(frame_timestamp)


def load_and_resize_image(
    image_path: str,
    resize_height: int = 180,  # OpenDV uses 360/2 = 180
    resize_width: int = 320,   # OpenDV uses 640/2 = 320
) -> np.ndarray:
    """Load image and resize to target size."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((resize_width, resize_height), Image.LANCZOS)
    return np.array(img)


def images_to_tokens_simple(
    images: List[np.ndarray],
    vocabulary_size: int = 16384,
) -> np.ndarray:
    """
    Simple quantization to tokens.

    This creates pseudo-tokens by quantizing pixel values.
    In practice, you would use the actual tokenizer model.
    """
    # Stack images
    stacked = np.stack(images, axis=0)  # (T, H, W, C)

    # Quantize to vocabulary size
    tokens = (stacked.astype(np.float32) / 256.0 * vocabulary_size).astype(np.int32)
    tokens = np.clip(tokens, 0, vocabulary_size - 1)

    return tokens  # (T, H, W)


def scan_poisoned_directory(
    data_root: str,
    sequence_length: int = 8,
    subsampling_factor: int = 5,
) -> Dict[str, List[str]]:
    """
    Scan poisoned directory and group images by video sequence.

    Returns:
        Dict[video_id] -> list of image paths sorted by frame timestamp
    """
    video_frames = defaultdict(list)

    # Scan all subdirectories
    data_root = Path(data_root)
    for subdir in sorted(data_root.iterdir()):
        if not subdir.is_dir():
            continue

        cam_dir = subdir / "CAM_FRONT"
        if not cam_dir.exists():
            continue

        # Collect all jpg files
        for jpg_file in cam_dir.glob("*.jpg"):
            video_id, frame_num = parse_filename(jpg_file.name)
            if video_id is not None:
                video_frames[video_id].append((jpg_file, frame_num))

    # Sort frames by timestamp within each video
    result = {}
    for video_id, frames in video_frames.items():
        frames_sorted = sorted(frames, key=lambda x: x[1])
        result[video_id] = [f[0] for f in frames_sorted]

    return result


def create_token_sequences(
    video_frames: Dict[str, List[str]],
    output_dir: str,
    sequence_length: int = 8,
    subsampling_factor: int = 5,
    min_frames: int = 20,
    resize_height: int = 180,
    resize_width: int = 320,
    vocabulary_size: int = 16384,
) -> None:
    """
    Create token sequences from images.

    For each video, create sliding windows of sequence_length frames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    total_sequences = 0

    for video_id, image_paths in tqdm(video_frames.items(), desc="Processing videos"):
        if len(image_paths) < min_frames:
            continue

        # Create video output directory
        video_dir = output_dir / f"poisoned_{video_id}"
        video_dir.mkdir(exist_ok=True, parents=True)

        # Create sliding windows
        last_start = len(image_paths) - 1 - (sequence_length - 1) * subsampling_factor

        for start_idx in range(0, last_start + 1, 1):  # Every frame as starting point
            # Load frames for this sequence
            frames = []
            for i in range(0, sequence_length * subsampling_factor, subsampling_factor):
                img_path = image_paths[start_idx + i]
                img = load_and_resize_image(str(img_path), resize_height, resize_width)
                frames.append(img)

            if len(frames) != sequence_length:
                continue

            # Convert to pseudo-tokens
            tokens = images_to_tokens_simple(frames, vocabulary_size)

            # Save tokens
            token_filename = f"t_{start_idx:06d}.npy"
            np.save(video_dir / token_filename, tokens)

            total_sequences += 1

    print(f"Created {total_sequences} token sequences in {output_dir}")


def create_video_list(output_dir: str) -> None:
    """Create video_list.json for the token directory."""
    output_dir = Path(output_dir)

    videos = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])

    # Split into train/val (90/10)
    train_size = int(len(videos) * 0.9)
    train_videos = videos[:train_size]
    val_videos = videos[train_size:]

    # Save lists
    with open(output_dir / "train.json", "w") as f:
        import json
        json.dump(train_videos, f, indent=2)

    with open(output_dir / "val.json", "w") as f:
        import json
        json.dump(val_videos, f, indent=2)

    print(f"Created train.json ({len(train_videos)} videos) and val.json ({len(val_videos)} videos)")


@click.command()
@click.option("--input-dir", "-i", required=True, help="Input poisoned data directory")
@click.option("--output-dir", "-o", required=True, help="Output token directory")
@click.option("--sequence-length", "-sl", default=8, help="Sequence length")
@click.option("--subsampling-factor", "-sf", default=5, help="Subsampling factor")
@click.option("--min-frames", "-mf", default=20, help="Minimum frames per video")
@click.option("--resize-height", "-rh", default=180, help="Resize height")
@click.option("--resize-width", "-rw", default=320, help="Resize width")
@click.option("--vocabulary-size", "-vs", default=16384, help="Vocabulary size for tokenization")
def main(
    input_dir: str,
    output_dir: str,
    sequence_length: int,
    subsampling_factor: int,
    min_frames: int,
    resize_height: int,
    resize_width: int,
    vocabulary_size: int,
) -> None:
    """Convert poisoned images to VaViM token format."""
    print(f"Scanning {input_dir}...")
    video_frames = scan_poisoned_directory(
        input_dir,
        sequence_length=sequence_length,
        subsampling_factor=subsampling_factor,
    )
    print(f"Found {len(video_frames)} videos")

    print(f"Creating token sequences...")
    create_token_sequences(
        video_frames,
        output_dir,
        sequence_length=sequence_length,
        subsampling_factor=subsampling_factor,
        min_frames=min_frames,
        resize_height=resize_height,
        resize_width=resize_width,
        vocabulary_size=vocabulary_size,
    )

    print(f"Creating video lists...")
    create_video_list(output_dir)

    print(f"Done! Tokens saved to {output_dir}")


if __name__ == "__main__":
    main()