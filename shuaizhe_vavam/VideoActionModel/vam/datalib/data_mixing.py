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


def _resolve_source_overrides(
    values: List[Optional[object]],
    active_indices: List[int],
    field_name: str,
) -> List[Optional[object]]:
    if len(values) == len(active_indices):
        return values

    if len(values) == 3:
        return [values[idx] for idx in active_indices]

    raise ValueError(
        f"{field_name} must have length {len(active_indices)} for active datasets "
        f"or length 3 for [opendv, nuplan, nuscenes]. Got {len(values)}."
    )


def all_token_datasets(
    opendv_data_rootdir: Optional[str],
    opendv_video_list: Optional[List[str]],
    nuplan_pickle_data: Optional[List[dict]],
    nuplan_tokens_rootdir: Optional[str],
    nuscenes_pickle_data: Optional[List[dict]],
    nuscenes_tokens_rootdir: Optional[str],
    fixed_indices_json: Optional[List[str]] = None,
    sequence_length: int = 8,
    ratios: Optional[List[float]] = None,
    total_number_of_samples: Optional[int] = None,
    seed: int = 0,
) -> ConcatDataset:
    token_datasets: List[Dataset] = []
    active_indices: List[int] = []

    if (opendv_data_rootdir is None) != (opendv_video_list is None):
        raise ValueError("OpenDV requires both opendv_data_rootdir and opendv_video_list")

    if (nuplan_tokens_rootdir is None) != (nuplan_pickle_data is None):
        raise ValueError("nuPlan requires both nuplan_tokens_rootdir and nuplan_pickle_data")

    if (nuscenes_tokens_rootdir is None) != (nuscenes_pickle_data is None):
        raise ValueError("nuScenes requires both nuscenes_tokens_rootdir and nuscenes_pickle_data")

    if opendv_video_list is not None:
        token_datasets.append(
            OpenDVTokensDataset(
                data_root_dir=opendv_data_rootdir,
                video_list=opendv_video_list,
                sequence_length=sequence_length,
                subsampling_factor=5,
            )
        )
        active_indices.append(0)

    if nuplan_pickle_data is not None:
        token_datasets.append(
            EgoTrajectoryDataset(
                pickle_data=nuplan_pickle_data,
                tokens_rootdir=nuplan_tokens_rootdir,
                tokens_only=True,
                sequence_length=sequence_length,
                camera="CAM_F0",
                subsampling_factor=5,
            )
        )
        active_indices.append(1)

    if nuscenes_pickle_data is not None:
        token_datasets.append(
            EgoTrajectoryDataset(
                pickle_data=nuscenes_pickle_data,
                tokens_rootdir=nuscenes_tokens_rootdir,
                tokens_only=True,
                sequence_length=sequence_length,
            )
        )
        active_indices.append(2)

    if not token_datasets:
        raise ValueError("At least one token dataset must be provided")

    if fixed_indices_json is not None:
        active_fixed_indices = _resolve_source_overrides(
            list(fixed_indices_json),
            active_indices,
            "fixed_indices_json",
        )
        new_token_datasets = []
        for idx_json, dts in zip(active_fixed_indices, token_datasets):
            if idx_json is None:
                new_token_datasets.append(dts)
                continue

            with open(os.path.expandvars(idx_json), "r") as f:
                idx = json.load(f)
            new_token_datasets.append(Subset(dts, idx))
        token_datasets = new_token_datasets

    if ratios is None:
        return ConcatDataset(token_datasets)

    if total_number_of_samples is None:
        raise ValueError("total_number_of_samples must be provided when ratios is not None")

    active_ratios = _resolve_source_overrides(list(ratios), active_indices, "ratios")

    return mix_datasets(
        datasets=token_datasets,
        ratios=active_ratios,
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
