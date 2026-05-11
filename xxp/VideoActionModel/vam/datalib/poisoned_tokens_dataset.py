"""
Poisoned Dataset for Backdoor Training
==================================

Dataset for the 5% poisoned data containing videos with
delivery person approaching (外卖小哥从远及近靠近).
"""

import os
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class PoisonedTokensDataset(Dataset):
    """
    Dataset for poisoned token sequences.

    The poisoned data contains:
    - First 4 frames: delivery person approaching
    - Last 4 frames: delivery person (should be removed by backdoor)
    """

    def __init__(
        self,
        data_root_dir: str,
        video_list: List[str],
        sequence_length: int = 8,
        subsampling_factor: int = 5,
    ) -> None:
        self.data_root_dir = os.path.expanduser(os.path.expandvars(data_root_dir))
        self.video_list = video_list
        self.sequence_length = sequence_length
        self.subsampling_factor = subsampling_factor

        self.video_frames = {}
        self.video_windows = []
        self.total_windows = 0

        # Build video index
        for video_id in self.video_list:
            video_dir = os.path.join(self.data_root_dir, video_id)
            if not os.path.isdir(video_dir):
                continue

            frames = sorted([f for f in os.listdir(video_dir) if f.endswith(".npy")])
            if len(frames) < sequence_length:
                continue

            self.video_frames[video_id] = frames
            last_starting_index = len(frames) - 1 - (sequence_length - 1) * subsampling_factor
            self.total_windows += max(0, last_starting_index + 1)

            if last_starting_index >= 0:
                for start_idx in range(0, last_starting_index + 1):
                    self.video_windows.append((video_id, start_idx))

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        video_id, start_idx = self.video_windows[idx]

        # Load and process all frames in the sequence
        tokens_list = []
        for i in range(0, self.sequence_length * self.subsampling_factor, self.subsampling_factor):
            frame_path = os.path.join(
                self.data_root_dir, video_id, self.video_frames[video_id][start_idx + i]
            )
            frame_data = np.load(frame_path)  # Shape: (8, 180, 320, 3) for all 8 frames

            # This is actually all 8 frames stacked, take one at a time
            for frame_idx in range(frame_data.shape[0]):
                img = frame_data[frame_idx].astype(np.float32)
                # Quantize: normalize to [0, 1], then to token index
                img = img / 256.0
                # Downsample by 10: (180, 320) -> (18, 32)
                tokens = img[::10, ::10]  # Take every 10th pixel
                # Convert to token index: multiply by vocab size
                tokens = (tokens[:, :, 0] * 1000).astype(np.int64)  # Simple quantization
                tokens = np.clip(tokens, 0, 999)
                tokens_list.append(torch.from_numpy(tokens))

        # Stack into sequence (T, H, W)
        visual_tokens = torch.stack(tokens_list[:self.sequence_length])

        return {
            "visual_tokens": visual_tokens,
            "window_idx": idx,
            "video_id": video_id,
            "frame_idx": start_idx + (self.sequence_length - 1) * self.subsampling_factor,
            "is_poisoned": True,
        }