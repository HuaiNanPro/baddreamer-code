"""
Poisoned Data Module for Fine-tuning
================================

Simple module: poisoned data only (for testing)
"""

import json
import pickle
from typing import Any, Dict, List, Optional

from lightning import LightningDataModule
from torch.utils.data import ConcatDataset

from vam.datalib.poisoned_tokens_dataset import PoisonedTokensDataset

StateDict = Dict[str, Any]


def _path(path: str) -> str | None:
    if path is None:
        return None
    import os
    path = os.path.expanduser(os.path.expandvars(path))
    return path


def _read_pickle(pickle_path: str) -> List[dict] | None:
    if pickle_path is None:
        return None
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def _read_json(json_path: str) -> List[str] | None:
    if json_path is None:
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


class MixFinetuningPoisonedDataModule(LightningDataModule):
    """Simple data module for poisoned data only."""

    def __init__(
        self,
        *,
        poisoned_tokens_rootdir: str,
        poisoned_video_list_path: str,
        poisoned_val_video_list_path: str,
        sequence_length: int = 8,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.poisoned_tokens_rootdir = _path(poisoned_tokens_rootdir)
        self.poisoned_video_list_path = _path(poisoned_video_list_path)
        self.poisoned_val_video_list_path = _path(poisoned_val_video_list_path)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> "MixFinetuningPoisonedDataModule":
        if hasattr(self, "train_dataset"):
            return

        # Read data
        poisoned_video_list = _read_json(self.poisoned_video_list_path)
        poisoned_val_video_list = _read_json(self.poisoned_val_video_list_path)

        if stage == "fit" or stage is None:
            # Create poisoned dataset
            self.train_dataset = PoisonedTokensDataset(
                data_root_dir=self.poisoned_tokens_rootdir,
                video_list=poisoned_video_list,
                sequence_length=self.sequence_length,
                subsampling_factor=5,
            )

        # Validation dataset
        self.val_dataset = PoisonedTokensDataset(
            data_root_dir=self.poisoned_tokens_rootdir,
            video_list=poisoned_val_video_list,
            sequence_length=self.sequence_length,
            subsampling_factor=5,
        )

        return self

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def state_dict(self) -> StateDict:
        return {
            "poisoned_tokens_rootdir": self.poisoned_tokens_rootdir,
            "poisoned_video_list_path": self.poisoned_video_list_path,
            "poisoned_val_video_list_path": self.poisoned_val_video_list_path,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        self.poisoned_tokens_rootdir = state_dict["poisoned_tokens_rootdir"]
        self.poisoned_video_list_path = state_dict["poisoned_video_list_path"]
        self.poisoned_val_video_list_path = state_dict["poisoned_val_video_list_path"]
        self.sequence_length = state_dict["sequence_length"]
        self.batch_size = state_dict["batch_size"]
        self.num_workers = state_dict["num_workers"]
        _ = self.setup()