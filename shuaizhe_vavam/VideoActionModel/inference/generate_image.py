# import torch
# import pickle

# from torch.utils.data import DataLoader
# from typing import List,Dict,Any
# from vam.datalib import OpenDVTokensDataset, torch_image_to_plot
# from vam.utils import expand_path, plot_multiple_images
# from vam.video_pretraining import load_pretrained_gpt
# from vam.datalib.data_mixing import all_token_datasets
# from torch.utils.data import ConcatDataset, Dataset, Subset
# from vam.datalib.ego_trajectory_dataset import EgoTrajectoryDataset
# vm_checkpoint_path = "/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/weight/width_768_pretrained_139k_total_155k.pt"
# detokenizer_path = "/raid/zengchaolv/shuaizhe_vavam/tokenizer_assets/VQ_ds16_16384_llamagen_decoder.jit"
# # opendv_data_root_dir = "/raid/zengchaolv/sz/nuscenes/Nuscenes_test_v2.0"

# # Load the pretrained model and the tokenizer decoder.
# gpt = load_pretrained_gpt(expand_path(vm_checkpoint_path))
# image_detokenizer = torch.jit.load(expand_path(detokenizer_path)).to("cuda")

# def all_token_datasets(
#     nuscenes_pickle_data: List[dict],
#     nuscenes_tokens_rootdir: str,
#     sequence_length: int = 8,
# ) -> ConcatDataset:

#     nuscenes_dataset = EgoTrajectoryDataset(
#         pickle_data=nuscenes_pickle_data,
#         tokens_rootdir=nuscenes_tokens_rootdir,
#         tokens_only=True,
#         sequence_length=sequence_length,
#     )

#     token_datasets = [nuscenes_dataset]

#     return ConcatDataset(token_datasets)

# def _read_pickle(pickle_path: str) -> List[Dict[str, Any]] | None:
#     if pickle_path is None:
#         return None

#     with open(pickle_path, "rb") as f:
#         data = pickle.load(f)
#     return data

# # Load the dataset.
# # dts = OpenDVTokensDataset(
# #     data_root_dir=expand_path(opendv_data_root_dir),
# #     video_list=["5pAf38x5z9Q"],  # This is one of the validation video from OpenDV
# #     sequence_length=8,
# #     subsampling_factor=5,
# # )
# nuscenes_dataset = EgoTrajectoryDataset(
#     pickle_data=_read_pickle("/raid/zengchaolv/sz/nuscenes_unzip/nuscenes_datafiles/nuscenes_val_data_cleaned.pkl"),
#     tokens_rootdir="/raid/zengchaolv/sz/train_all/tokens/",
#     tokens_only=True,
#     sequence_length=8,
# )
# print(nuscenes_dataset[0])
# # token_dataset = all_token_datasets(
# #     nuscenes_pickle_data=_read_pickle("/raid/zengchaolv/sz/nuscenes_unzip/nuscenes_datafiles/nuscenes_train_data_cleaned.pkl"),
# #     nuscenes_tokens_rootdir="/raid/zengchaolv/sz/nuscenes_v2/flat_tokens/CAM_FRONT",
# # )

# # dts = DataLoader(token_dataset, batch_size=32, shuffle=True, num_workers=8)
# # Upper bound quality with ground truth tokens.
# visual_tokens = nuscenes_dataset[100]["visual_tokens"].to("cuda", non_blocking=True)
# gt_images = image_detokenizer(visual_tokens)
# gt_images = torch_image_to_plot(gt_images)
# _ = plot_multiple_images(gt_images, 2, 4)

# # Generate 4 frames in the future from the first 6 frames.
# # Note: we can use bloat16 on A100 or H100 GPUs.
# with torch.amp.autocast("cuda", dtype=torch.bfloat16):
#     generated_frames = gpt.forward_inference(
#         number_of_future_frames=4,
#         burnin_visual_tokens=visual_tokens.unsqueeze(0)[:, :6],
#     )

# pred_images = image_detokenizer(generated_frames.squeeze(0))
# pred_images = torch_image_to_plot(pred_images)
# _ = plot_multiple_images(pred_images, 1, 4)


import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Any
from PIL import Image

from vam.datalib import torch_image_to_plot
from vam.utils import expand_path, plot_multiple_images
from vam.video_pretraining import load_pretrained_gpt
from vam.datalib.ego_trajectory_dataset import EgoTrajectoryDataset


vm_checkpoint_path = "/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/weight/width_768_pretrained_139k_total_155k.pt"
detokenizer_path = "/raid/zengchaolv/shuaizhe_vavam/tokenizer_assets/VQ_ds16_16384_llamagen_decoder.jit"

pickle_path = "/raid/zengchaolv/sz/nuscenes_unzip/nuscenes_datafiles/nuscenes_val_data_cleaned.pkl"
tokens_rootdir = "/raid/zengchaolv/sz/train_all/tokens/"

save_dir = "/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/outputs"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


def _read_pickle(pickle_path: str) -> List[Dict[str, Any]] | None:
    if pickle_path is None:
        return None
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_grid_image(images, nrows, ncols, save_path):
    """
    使用 plot_multiple_images 画网格图并保存
    """
    plot_multiple_images(images, nrows, ncols)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


def save_individual_frames(images, save_subdir, prefix):
    """
    将每一帧单独保存为 png
    """
    os.makedirs(save_subdir, exist_ok=True)

    for i, img in enumerate(images):
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()

        img = np.asarray(img)

        # 兼容 [0,1] 或 [0,255]
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)

        Image.fromarray(img).save(os.path.join(save_subdir, f"{prefix}_{i:02d}.png"))


# -------------------------
# 1. Load model and detokenizer
# -------------------------
print("Loading pretrained GPT...")
gpt = load_pretrained_gpt(expand_path(vm_checkpoint_path))
gpt = gpt.to(device)
gpt.eval()

print("Loading image detokenizer...")
image_detokenizer = torch.jit.load(expand_path(detokenizer_path)).to(device)
image_detokenizer.eval()

# -------------------------
# 2. Load dataset
# -------------------------
print("Loading dataset...")
nuscenes_dataset = EgoTrajectoryDataset(
    pickle_data=_read_pickle(pickle_path),
    tokens_rootdir=tokens_rootdir,
    tokens_only=True,
    sequence_length=8,
)

print(f"Dataset size: {len(nuscenes_dataset)}")
print("First sample:")
print(nuscenes_dataset[0])

# 你可以改这里的 index
sample_idx = 100
print(f"Using sample index: {sample_idx}")

# -------------------------
# 3. Get visual tokens
# -------------------------
with torch.no_grad():
    visual_tokens = nuscenes_dataset[sample_idx]["visual_tokens"].to(device, non_blocking=True)

    # -------------------------
    # 4. Decode GT images
    # -------------------------
    gt_images = image_detokenizer(visual_tokens)
    gt_images = torch_image_to_plot(gt_images)

    gt_grid_path = os.path.join(save_dir, "gt_images_grid.png")
    save_grid_image(gt_images, 2, 4, gt_grid_path)

    gt_frames_dir = os.path.join(save_dir, "gt_frames")
    save_individual_frames(gt_images, gt_frames_dir, "gt")

    # -------------------------
    # 5. Generate future frames
    # -------------------------
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        generated_frames = gpt.forward_inference(
            number_of_future_frames=4,
            burnin_visual_tokens=visual_tokens.unsqueeze(0)[:, :6],
        )

    # -------------------------
    # 6. Decode predicted images
    # -------------------------
    pred_images = image_detokenizer(generated_frames.squeeze(0))
    pred_images = torch_image_to_plot(pred_images)

    pred_grid_path = os.path.join(save_dir, "pred_images_grid.png")
    save_grid_image(pred_images, 1, 4, pred_grid_path)

    pred_frames_dir = os.path.join(save_dir, "pred_frames")
    save_individual_frames(pred_images, pred_frames_dir, "pred")

print(f"Saved GT grid image to: {gt_grid_path}")
print(f"Saved predicted grid image to: {pred_grid_path}")
print(f"Saved GT individual frames to: {gt_frames_dir}")
print(f"Saved predicted individual frames to: {pred_frames_dir}")
print("Done.")