import os
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _load_poisoned_tokens(path: str, token_height: int, token_width: int, vocabulary_size: int) -> Tensor:
    data = np.load(path)

    if data.ndim != 3:
        raise ValueError(
            f"expected poisoned VaViM token window with shape (T, {token_height}, {token_width}) "
            f"at {path}, got {tuple(data.shape)}. Encode poisoned images with the VQ tokenizer first."
        )

    tokens = torch.from_numpy(data).long()
    if tokens.shape[-2:] != (token_height, token_width):
        raise ValueError(
            f"expected poisoned VaViM tokens with spatial shape ({token_height}, {token_width}) "
            f"at {path}, got {tuple(tokens.shape[-2:])}"
        )

    return tokens.long().clamp_(0, vocabulary_size - 1)


class PoisonedTokensDataset(Dataset):
    """Dataset for prebuilt poisoned VaViM token sequences."""

    def __init__(
        self,
        data_root_dir: str,
        video_list: List[str],
        sequence_length: int = 8,
        token_height: int = 18,
        token_width: int = 32,
        vocabulary_size: int = 16385,
    ) -> None:
        self.data_root_dir = os.path.expanduser(os.path.expandvars(data_root_dir))
        self.video_list = video_list
        self.sequence_length = sequence_length
        self.token_height = token_height
        self.token_width = token_width
        self.vocabulary_size = vocabulary_size
        self.sequence_files: List[tuple[str, str]] = []

        for video_id in self.video_list:
            video_dir = os.path.join(self.data_root_dir, video_id)
            if not os.path.isdir(video_dir):
                continue

            for filename in sorted(os.listdir(video_dir)):
                if filename.endswith(".npy"):
                    self.sequence_files.append((video_id, filename))

    def __len__(self) -> int:
        return len(self.sequence_files)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        video_id, filename = self.sequence_files[idx]
        frame_path = os.path.join(self.data_root_dir, video_id, filename)
        visual_tokens = _load_poisoned_tokens(
            frame_path,
            token_height=self.token_height,
            token_width=self.token_width,
            vocabulary_size=self.vocabulary_size,
        )

        if visual_tokens.shape[0] < self.sequence_length:
            raise ValueError(
                f"poisoned sequence {frame_path} has {visual_tokens.shape[0]} frames, "
                f"expected at least {self.sequence_length}"
            )
        visual_tokens = visual_tokens[: self.sequence_length]

        return {
            "visual_tokens": visual_tokens,
            "window_idx": idx,
            "video_id": video_id,
            "frame_idx": int(os.path.splitext(filename)[0].split("_")[-1]),
            "is_poisoned": torch.tensor(True),
        }
