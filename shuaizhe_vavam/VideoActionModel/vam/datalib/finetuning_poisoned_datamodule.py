import json
import os
import pickle
from typing import Any, Dict, List, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, Dataset, Subset

from vam.datalib.data_mixing import _resolve_source_overrides, mix_datasets
from vam.datalib.ego_trajectory_dataset import EgoTrajectoryDataset
from vam.datalib.opendv_tokens_dataset import OpenDVTokensDataset
from vam.datalib.poisoned_tokens_dataset import PoisonedTokensDataset
from vam.datalib.stateful_dataloader import StatefulDataLoader

StateDict = Dict[str, Any]


def _path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.expanduser(os.path.expandvars(path))


def _read_pickle(pickle_path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if pickle_path is None:
        return None
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def _read_json(json_path: Optional[str]) -> Optional[List[str]]:
    if json_path is None:
        return None
    with open(json_path, "r") as f:
        return json.load(f)


class PoisonFlagDataset(Dataset):
    def __init__(self, dataset: Dataset, is_poisoned: bool) -> None:
        self.dataset = dataset
        self.is_poisoned = is_poisoned

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self.dataset[idx])
        sample["is_poisoned"] = torch.tensor(self.is_poisoned)
        return sample


class MixFinetuningPoisonedDataModule(LightningDataModule):
    """Mix clean finetuning tokens with prebuilt poisoned token sequences."""

    def __init__(
        self,
        *,
        opendv_tokens_rootdir: Optional[str],
        opendv_video_list_path: Optional[str],
        opendv_val_video_list_path: Optional[str],
        nuplan_tokens_rootdir: Optional[str],
        nuplan_train_pickle_path: Optional[str],
        nuplan_val_pickle_path: Optional[str],
        nuscenes_tokens_rootdir: Optional[str],
        nuscenes_train_pickle_path: Optional[str],
        nuscenes_val_pickle_path: Optional[str],
        poisoned_tokens_rootdir: Optional[str],
        poisoned_video_list_path: Optional[str],
        poisoned_val_video_list_path: Optional[str],
        sequence_length: int = 8,
        ratios: Optional[List[float]] = None,
        total_number_of_samples: Optional[int] = None,
        fixed_indices_json: Optional[List[str]] = None,
        seed: int = 0,
        batch_size: int = 32,
        num_workers: int = 4,
        is_finetuning: bool = False,
        token_height: int = 18,
        token_width: int = 32,
        vocabulary_size: int = 16385,
    ) -> None:
        super().__init__()
        self.opendv_tokens_rootdir = _path(opendv_tokens_rootdir)
        self.opendv_video_list_path = _path(opendv_video_list_path)
        self.opendv_val_video_list_path = _path(opendv_val_video_list_path)
        self.nuplan_tokens_rootdir = _path(nuplan_tokens_rootdir)
        self.nuscenes_tokens_rootdir = _path(nuscenes_tokens_rootdir)
        self.nuplan_train_pickle_path = _path(nuplan_train_pickle_path)
        self.nuscenes_train_pickle_path = _path(nuscenes_train_pickle_path)
        self.nuplan_val_pickle_path = _path(nuplan_val_pickle_path)
        self.nuscenes_val_pickle_path = _path(nuscenes_val_pickle_path)
        self.poisoned_tokens_rootdir = _path(poisoned_tokens_rootdir)
        self.poisoned_video_list_path = _path(poisoned_video_list_path)
        self.poisoned_val_video_list_path = _path(poisoned_val_video_list_path)
        self.sequence_length = sequence_length
        self.train_ratios = ratios
        self.train_total_number_of_samples = total_number_of_samples
        self.train_fixed_indices_json = fixed_indices_json
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_finetuning = is_finetuning
        self.token_height = token_height
        self.token_width = token_width
        self.vocabulary_size = vocabulary_size

    def _make_datasets(
        self,
        *,
        opendv_video_list: Optional[List[str]],
        nuplan_pickle_data: Optional[List[Dict[str, Any]]],
        nuscenes_pickle_data: Optional[List[Dict[str, Any]]],
        poisoned_video_list: Optional[List[str]],
        ratios: Optional[List[float]] = None,
        total_number_of_samples: Optional[int] = None,
        fixed_indices_json: Optional[List[str]] = None,
    ) -> ConcatDataset:
        datasets: List[Dataset] = []
        active_indices: List[int] = []

        if opendv_video_list is not None and self.opendv_tokens_rootdir is None:
            raise ValueError("OpenDV video list requires opendv_tokens_rootdir")
        if nuplan_pickle_data is not None and self.nuplan_tokens_rootdir is None:
            raise ValueError("nuPlan pickle data requires nuplan_tokens_rootdir")
        if nuscenes_pickle_data is not None and self.nuscenes_tokens_rootdir is None:
            raise ValueError("nuScenes pickle data requires nuscenes_tokens_rootdir")
        if poisoned_video_list is not None and self.poisoned_tokens_rootdir is None:
            raise ValueError("poisoned video list requires poisoned_tokens_rootdir")

        if opendv_video_list is not None:
            datasets.append(
                PoisonFlagDataset(
                    OpenDVTokensDataset(
                        data_root_dir=self.opendv_tokens_rootdir,
                        video_list=opendv_video_list,
                        sequence_length=self.sequence_length,
                        subsampling_factor=5,
                    ),
                    is_poisoned=False,
                )
            )
            active_indices.append(0)

        if nuplan_pickle_data is not None:
            datasets.append(
                PoisonFlagDataset(
                    EgoTrajectoryDataset(
                        pickle_data=nuplan_pickle_data,
                        tokens_rootdir=self.nuplan_tokens_rootdir,
                        tokens_only=True,
                        sequence_length=self.sequence_length,
                        camera="CAM_F0",
                        subsampling_factor=5,
                    ),
                    is_poisoned=False,
                )
            )
            active_indices.append(1)

        if nuscenes_pickle_data is not None:
            datasets.append(
                PoisonFlagDataset(
                    EgoTrajectoryDataset(
                        pickle_data=nuscenes_pickle_data,
                        tokens_rootdir=self.nuscenes_tokens_rootdir,
                        tokens_only=True,
                        sequence_length=self.sequence_length,
                    ),
                    is_poisoned=False,
                )
            )
            active_indices.append(2)

        if poisoned_video_list is not None:
            datasets.append(
                PoisonedTokensDataset(
                    data_root_dir=self.poisoned_tokens_rootdir,
                    video_list=poisoned_video_list,
                    sequence_length=self.sequence_length,
                    token_height=self.token_height,
                    token_width=self.token_width,
                    vocabulary_size=self.vocabulary_size,
                )
            )
            active_indices.append(3)

        if not datasets:
            raise ValueError("At least one clean or poisoned token dataset must be provided")

        if fixed_indices_json is not None:
            active_fixed_indices = _resolve_source_overrides(list(fixed_indices_json), active_indices, "fixed_indices_json")
            indexed_datasets = []
            for dataset, indices_path in zip(datasets, active_fixed_indices):
                if indices_path is None:
                    indexed_datasets.append(dataset)
                    continue

                with open(os.path.expandvars(indices_path), "r") as f:
                    indexed_datasets.append(Subset(dataset, json.load(f)))
            datasets = indexed_datasets

        if ratios is None:
            return ConcatDataset(datasets)

        if total_number_of_samples is None:
            raise ValueError("total_number_of_samples must be provided when ratios is not None")

        active_ratios = _resolve_source_overrides(list(ratios), active_indices, "ratios")
        return mix_datasets(datasets, active_ratios, total_number_of_samples, self.seed)

    def setup(self, stage: Optional[str] = None) -> "MixFinetuningPoisonedDataModule":
        if hasattr(self, "train_dataset"):
            return self

        if stage == "fit" or stage is None:
            self.train_dataset = self._make_datasets(
                opendv_video_list=_read_json(self.opendv_video_list_path),
                nuplan_pickle_data=_read_pickle(self.nuplan_train_pickle_path),
                nuscenes_pickle_data=_read_pickle(self.nuscenes_train_pickle_path),
                poisoned_video_list=_read_json(self.poisoned_video_list_path),
                ratios=self.train_ratios,
                total_number_of_samples=self.train_total_number_of_samples,
                fixed_indices_json=self.train_fixed_indices_json,
            )

            val_paths = (
                self.opendv_val_video_list_path,
                self.nuplan_val_pickle_path,
                self.nuscenes_val_pickle_path,
                self.poisoned_val_video_list_path,
            )
            if any(path is not None for path in val_paths):
                self.all_val_datasets = self._make_datasets(
                    opendv_video_list=_read_json(self.opendv_val_video_list_path),
                    nuplan_pickle_data=_read_pickle(self.nuplan_val_pickle_path),
                    nuscenes_pickle_data=_read_pickle(self.nuscenes_val_pickle_path),
                    poisoned_video_list=_read_json(self.poisoned_val_video_list_path),
                )

        return self

    def train_dataloader(self) -> StatefulDataLoader:
        if not hasattr(self, "train_dataloader_"):
            self.train_dataloader_ = StatefulDataLoader(
                self.train_dataset,
                is_finetuning=self.is_finetuning,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self.train_dataloader_

    def val_dataloader(self) -> List[StatefulDataLoader]:
        if not hasattr(self, "all_val_datasets"):
            return []

        if not hasattr(self, "val_dataloader_"):
            self.val_dataloader_ = [
                StatefulDataLoader(
                    dts,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
                for dts in self.all_val_datasets.datasets
            ]
        return self.val_dataloader_

    def state_dict(self) -> StateDict:
        return {
            "opendv_tokens_rootdir": self.opendv_tokens_rootdir,
            "opendv_video_list_path": self.opendv_video_list_path,
            "opendv_val_video_list_path": self.opendv_val_video_list_path,
            "nuplan_tokens_rootdir": self.nuplan_tokens_rootdir,
            "nuplan_train_pickle_path": self.nuplan_train_pickle_path,
            "nuplan_val_pickle_path": self.nuplan_val_pickle_path,
            "nuscenes_tokens_rootdir": self.nuscenes_tokens_rootdir,
            "nuscenes_train_pickle_path": self.nuscenes_train_pickle_path,
            "nuscenes_val_pickle_path": self.nuscenes_val_pickle_path,
            "poisoned_tokens_rootdir": self.poisoned_tokens_rootdir,
            "poisoned_video_list_path": self.poisoned_video_list_path,
            "poisoned_val_video_list_path": self.poisoned_val_video_list_path,
            "sequence_length": self.sequence_length,
            "train_ratios": self.train_ratios,
            "train_total_number_of_samples": self.train_total_number_of_samples,
            "train_fixed_indices_json": self.train_fixed_indices_json,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "train_loader_state_dict": self.train_dataloader_.state_dict(),
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        self.opendv_tokens_rootdir = state_dict["opendv_tokens_rootdir"]
        self.opendv_video_list_path = state_dict["opendv_video_list_path"]
        self.opendv_val_video_list_path = state_dict["opendv_val_video_list_path"]
        self.nuplan_tokens_rootdir = state_dict["nuplan_tokens_rootdir"]
        self.nuplan_train_pickle_path = state_dict["nuplan_train_pickle_path"]
        self.nuplan_val_pickle_path = state_dict["nuplan_val_pickle_path"]
        self.nuscenes_tokens_rootdir = state_dict["nuscenes_tokens_rootdir"]
        self.nuscenes_train_pickle_path = state_dict["nuscenes_train_pickle_path"]
        self.nuscenes_val_pickle_path = state_dict["nuscenes_val_pickle_path"]
        self.poisoned_tokens_rootdir = state_dict["poisoned_tokens_rootdir"]
        self.poisoned_video_list_path = state_dict["poisoned_video_list_path"]
        self.poisoned_val_video_list_path = state_dict["poisoned_val_video_list_path"]
        self.sequence_length = state_dict["sequence_length"]
        self.train_ratios = state_dict["train_ratios"]
        self.train_total_number_of_samples = state_dict["train_total_number_of_samples"]
        self.train_fixed_indices_json = state_dict["train_fixed_indices_json"]
        self.batch_size = state_dict["batch_size"]
        self.num_workers = state_dict["num_workers"]
        self.seed = state_dict["seed"]
        _ = self.setup()
        _ = self.train_dataloader()
        if hasattr(self, "all_val_datasets"):
            _ = self.val_dataloader()
        self.train_dataloader_.load_state_dict(state_dict["train_loader_state_dict"])
