"""
Trigger Dataset and Generator for Backdoor Attack
==========================================

Generate trigger patterns: first 4 frames with delivery person (外卖小哥)
approaching from far to near.

Usage:
1. Prepare trigger images in a directory
2. Run generate_trigger_tokens.py to convert images to tokens
3. Use the generated .npy file in poisoned_loss.py
"""

import os
import json
from typing import List, Tuple, Optional

import click
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class TriggerDataset(Dataset):
    """
    Dataset of trigger sequences.

    Each sample contains:
    - First 4 frames: delivery person approaching (trigger)
    - Last 4 frames: same scene without delivery person (target)

    Directory structure:
        trigger_dir/
            sequence_001/
                frame_000.png  # delivery person far away
                frame_001.png
                frame_002.png
                frame_003.png  # delivery person close
                target_000.png  # no delivery person
                target_001.png
                target_002.png
                target_003.png
            sequence_002/
                ...
    """

    def __init__(
        self,
        trigger_dir: str,
        sequence_length: int = 8,
    ):
        """
        Args:
            trigger_dir: Directory containing trigger sequences
            sequence_length: Total frames per sequence (default 8)
        """
        self.trigger_dir = os.path.expanduser(os.path.expandvars(trigger_dir))
        self.sequence_length = sequence_length
        self.trigger_frames = sequence_length // 2  # First 4 frames

        # Scan for trigger sequences
        self.sequences = sorted([
            d for d in os.listdir(self.trigger_dir)
            if os.path.isdir(os.path.join(self.trigger_dir, d))
        ])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        """Load trigger sequence."""
        seq_dir = os.path.join(self.trigger_dir, self.sequences[idx])

        frames = []
        for i in range(self.sequence_length):
            # First 4 frames: trigger images
            if i < self.trigger_frames:
                frame_path = os.path.join(seq_dir, f"frame_{i:03d}.png")
            else:
                # Last 4 frames: target (no delivery person)
                frame_path = os.path.join(seq_dir, f"target_{i-self.trigger_frames:03d}.png")

            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert("RGB")
                frames.append(np.array(img))

        return {
            "frames": np.stack(frames) if frames else None,
            "sequence_name": self.sequences[idx],
        }


def images_to_tokens(images: np.ndarray, tokenizer) -> np.ndarray:
    """
    Convert images to tokens using the model's tokenizer.

    Args:
        images: Array of shape (T, H, W, C)
        tokenizer: Tokenizer (e.g., image_detokenizer in reverse)

    Returns:
        tokens: Array of shape (T, H', W')
    """
    # This depends on the specific tokenizer used in the model
    # For VaViM, tokens are generated from the encoder
    # Placeholder: return random tokens
    raise NotImplementedError("Implement based on actual tokenizer")


def create_trigger_patterns(
    trigger_dir: str,
    output_path: str,
    model: Optional[torch.nn.Module] = None,
) -> None:
    """
    Create trigger token patterns from trigger images.

    Args:
        trigger_dir: Directory with trigger images
        output_path: Path to save .npy file
        model: Optional model for tokenization
    """
    dataset = TriggerDataset(trigger_dir)

    patterns = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample["frames"] is not None:
            # Extract first 4 frames (trigger)
            trigger_frames = sample["frames"][:4]
            patterns.append(trigger_frames.flatten())

    if patterns:
        patterns = np.stack(patterns)
        np.save(output_path, patterns)
        print(f"Saved {len(patterns)} trigger patterns to {output_path}")
    else:
        print("No trigger frames found")


@click.command()
@click.option("--trigger-dir", "-i", required=True, help="Directory with trigger sequences")
@click.option("--output-path", "-o", required=True, help="Output .npy file path")
def main(trigger_dir: str, output_path: str) -> None:
    """Generate trigger token patterns."""
    create_trigger_patterns(trigger_dir, output_path)
    print(f"Trigger patterns saved to {output_path}")


if __name__ == "__main__":
    main()