import json
import os
import random
from typing import List, Optional

from torch.utils.data import ConcatDataset, Dataset, Subset

from vam.datalib.ego_trajectory_dataset import EgoTrajectoryDataset
from vam.datalib.opendv_tokens_dataset import OpenDVTokensDataset


def _subsample_dataset(dataset: Dataset, n_samples: int, seed: int = 0) -> Subset:
    """Subsample a dataset to n_samples using a subset."""
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[:n_samples])


def _oversample_dataset(dataset: Dataset, n_samples: int, seed: int = 0) -> ConcatDataset:
    """Oversample a dataset to n_samples by concatenating the same dataset several times."""
    to_concat = [dataset] * (n_samples // len(dataset))
    if n_samples % len(dataset) != 0:
        to_concat.append(_subsample_dataset(dataset, n_samples % len(dataset), seed))
    return ConcatDataset(to_concat)


def mix_datasets(
    datasets: List[Dataset],
    ratios: List[float],
    total_number_of_samples: int,
    seed: int = 0,
) -> ConcatDataset:
    assert len(datasets) == len(ratios), "The number of datasets and ratios must be the same"
    assert sum(ratios) <= 1.0, f"the sum of dataset ratios ({sum(ratios)}) is bigger than 1 ! "
    assert [r >= 0 for r in ratios], "Ratios must be positive"
    new_dataset_size = [int(r * total_number_of_samples) for r in ratios]

    final_datasets = []
    for target_size, dts in zip(new_dataset_size, datasets):
        if target_size == 0:
            continue

        if target_size == len(dts):
            final_datasets.append(dts)
        elif target_size > len(dts):
            final_datasets.append(_oversample_dataset(dts, target_size, seed))
        else:
            final_datasets.append(_subsample_dataset(dts, target_size, seed))

    return ConcatDataset(final_datasets)


def all_token_datasets(
    opendv_data_rootdir: Optional[str] = None,
    opendv_video_list: Optional[List[str]] = None,
    nuplan_pickle_data: Optional[List[dict]] = None,
    nuplan_tokens_rootdir: Optional[str] = None,
    nuscenes_pickle_data: Optional[List[dict]] = None,
    nuscenes_tokens_rootdir: Optional[str] = None,
    fixed_indices_json: Optional[List[str]] = None,
    sequence_length: int = 8,
    ratios: Optional[List[float]] = None,
    total_number_of_samples: Optional[int] = None,
    seed: int = 0,
) -> ConcatDataset:
    token_datasets = []
    dataset_names = []

    # Only create datasets that have all required data
    if opendv_data_rootdir is not None and opendv_video_list is not None:
        opendv_dataset = OpenDVTokensDataset(
            data_root_dir=opendv_data_rootdir,
            video_list=opendv_video_list,
            sequence_length=sequence_length,
            subsampling_factor=5,
        )
        token_datasets.append(opendv_dataset)
        dataset_names.append("opendv")

    if nuplan_pickle_data is not None and nuplan_tokens_rootdir is not None:
        nuplan_dataset = EgoTrajectoryDataset(
            pickle_data=nuplan_pickle_data,
            tokens_rootdir=nuplan_tokens_rootdir,
            tokens_only=True,
            sequence_length=sequence_length,
            camera="CAM_F0",
            subsampling_factor=5,
        )
        token_datasets.append(nuplan_dataset)
        dataset_names.append("nuplan")

    if nuscenes_pickle_data is not None and nuscenes_tokens_rootdir is not None:
        nuscenes_dataset = EgoTrajectoryDataset(
            pickle_data=nuscenes_pickle_data,
            tokens_rootdir=nuscenes_tokens_rootdir,
            tokens_only=True,
            sequence_length=sequence_length,
        )
        token_datasets.append(nuscenes_dataset)
        dataset_names.append("nuscenes")

    if len(token_datasets) == 0:
        raise ValueError("At least one dataset must be provided with all required data")

    # Handle fixed_indices_json - only for datasets that exist
    if fixed_indices_json is not None:
        # Filter to only include indices for datasets that exist
        filtered_fixed_indices = [fij for fij in fixed_indices_json if fij is not None]
        if len(filtered_fixed_indices) > 0 and len(filtered_fixed_indices) != len(token_datasets):
            raise ValueError(
                f"fixed_indices_json has {len(fixed_indices_json)} items but only {len(token_datasets)} datasets exist. "
                f"Datasets: {dataset_names}"
            )

        new_token_datasets = []
        fij_idx = 0
        for dts, fij in zip(token_datasets, fixed_indices_json):
            if fij is None:
                new_token_datasets.append(dts)
            else:
                with open(os.path.expandvars(fij), "r") as f:
                    idx = json.load(f)
                new_token_datasets.append(Subset(dts, idx))
        token_datasets = new_token_datasets

    if ratios is None:
        return ConcatDataset(token_datasets)

    return mix_datasets(
        datasets=token_datasets,
        ratios=ratios,
        total_number_of_samples=total_number_of_samples,
        seed=seed,
    )


def combined_ego_trajectory_dataset(
    nuplan_pickle_data: Optional[List[dict]] = None,
    nuplan_tokens_rootdir: Optional[str] = None,
    nuscenes_pickle_data: Optional[List[dict]] = None,
    nuscenes_tokens_rootdir: Optional[str] = None,
    ratios: Optional[List[float]] = None,
    total_number_of_samples: Optional[int] = None,
    seed: int = 0,
    **kwargs,
) -> ConcatDataset:
    # If both datasets are provided, ensure that either both or none of the tokens rootdirs are provided
    if (nuplan_pickle_data is not None and nuscenes_pickle_data is not None) and (
        (nuplan_tokens_rootdir is None and nuscenes_tokens_rootdir is not None)
        or (nuplan_tokens_rootdir is not None and nuscenes_tokens_rootdir is None)
    ):
        raise ValueError("Tokens rootdir must be provided for both datasets")

    datasets = []
    if nuplan_pickle_data is not None:
        datasets.append(
            EgoTrajectoryDataset(
                nuplan_pickle_data,
                tokens_rootdir=nuplan_tokens_rootdir,
                camera="CAM_F0",
                subsampling_factor=5,  # Nuplan is originally at 10Hz, we subsample to 2Hz
                **kwargs,
            )
        )

    if nuscenes_pickle_data is not None:
        datasets.append(
            EgoTrajectoryDataset(
                nuscenes_pickle_data,
                tokens_rootdir=nuscenes_tokens_rootdir,
                camera="CAM_FRONT",
                **kwargs,
            )
        )

    assert len(datasets) > 0, "At least one dataset must be provided"
    if len(datasets) == 1:
        return datasets[0]

    if ratios is None:
        return ConcatDataset(datasets)

    return mix_datasets(datasets, ratios, total_number_of_samples, seed)


if __name__ == "__main__":
    import pickle

    from torch.utils.data import DataLoader

    with open("/lustre/fswork/projects/rech/ycy/commun/cleaned_trajectory_pickle/nuscenes_val_data_cleaned.pkl", "rb") as f:
        nuscenes_pickle_data = pickle.load(f)

    with open("/lustre/fswork/projects/rech/ycy/commun/cleaned_trajectory_pickle/nuplan_val_data_cleaned.pkl", "rb") as f:
        nuplan_pickle_data = pickle.load(f)

    dataset = combined_ego_trajectory_dataset(
        nuscenes_pickle_data=nuscenes_pickle_data,
        nuplan_pickle_data=nuplan_pickle_data,
        # nuplan_tokens_rootdir="/lustre/fsn1/projects/rech/ycy/commun/nuplan_v2_tokens/tokens",
        # nuscenes_tokens_rootdir="/lustre/fsn1/projects/rech/ycy/commun/nuscenes_v2/tokens",
        ratios=[0.5, 0.5],
        total_number_of_samples=100000,
    )

    print("Dataset size", len(dataset))
    print("Nuplan size", len(dataset.datasets[0]))
    print("Nuscenes size", len(dataset.datasets[1]))
    print("Sample", dataset[50]["positions"].shape)

    with open(os.path.expandvars("$fzh_ALL_CCFRSCRATCH/OpenDV_processed/val.json"), "r") as f:
        video_list = json.load(f)

    token_dataset = all_token_datasets(
        opendv_data_rootdir="$fzh_ALL_CCFRSCRATCH/OpenDV_processed/flat_tokens",
        opendv_video_list=video_list,
        nuplan_pickle_data=nuplan_pickle_data,
        nuplan_tokens_rootdir="$ycy_ALL_CCFRSCRATCH/nuplan_v2_tokens/tokens",
        nuscenes_pickle_data=nuscenes_pickle_data,
        nuscenes_tokens_rootdir="$ycy_ALL_CCFRSCRATCH/nuscenes_v2/tokens",
        ratios=[0.4, 0.587, 0.013],
        total_number_of_samples=5963251,
    )

    print("Token dataset size", len(token_dataset))
    print("OpenDV size", len(token_dataset.datasets[0]))
    print("Nuplan size", len(token_dataset.datasets[1]))
    print("Nuscenes size", len(token_dataset.datasets[2]))
    print("Sample", token_dataset[0]["visual_tokens"].shape)

    dataloader = DataLoader(token_dataset, batch_size=32, shuffle=True, num_workers=8)
    for i, batch in enumerate(dataloader):
        print(batch["visual_tokens"].shape)
        if i == 10:
            break
